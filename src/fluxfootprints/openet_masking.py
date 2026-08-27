"""
openet_masking.py
=================

Mask footprint output -- both the in-memory xarray fields and the exported
GeoTIFFs -- with the valid-data masks carried by daily OpenET rasters.

OpenET distributes daily ET as one raster per day with the date in the file
name (``ensemble_et_20200615.tif``, ``2020-06-15_openet.tif``, ...).  Cells with
no retrievable ET value -- outside the modelled crop mask, cloud screened, or
off-tile -- are stored as nodata.  Footprint weight that lands on those cells
cannot be paired with an ET value, so it should normally be dropped before the
footprint is used to attribute flux to the landscape.

The helpers here reproject that valid/invalid pattern from each daily OpenET
raster onto the footprint grid, apply it per time step (daily slices use the
matching day; monthly slices combine every OpenET day in that month), and report
the fraction of footprint weight that survives so the loss stays visible.

Typical use::

    model = build_climatology(df, ...)
    summaries = summarize_periods(model, df)

    masked = mask_summaries(summaries, "/data/openet/daily",
                            station_lat=40.0, station_lon=-111.9)

    export_rasters_geotiff(model, masked, 40.0, -111.9, out_dir="out")

or, to mask GeoTIFFs that were already written to disk::

    mask_rasters_geotiff("out", "/data/openet/daily")
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from .ffp_daily_monthly_helper import SummaryResult, _choose_utm_epsg_pyproj

__all__ = [
    "GridGeometry",
    "MaskedFootprint",
    "apply_openet_mask",
    "footprint_grid_geometry",
    "index_openet_rasters",
    "mask_footprint_dataarray",
    "mask_rasters_geotiff",
    "mask_summaries",
    "openet_mask_on_grid",
    "parse_raster_date",
]

# Anything convertible into a set of dated OpenET rasters.
OpenETSource = (
    str | Path | Sequence[str | Path] | Mapping[Any, str | Path | Sequence[str | Path]]
)

_DAY_RE = re.compile(r"(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)")
_MONTH_RE = re.compile(r"(?<!\d)(\d{4})[-_.]?(\d{2})(?!\d)")


# ------------------------------
# File-name date parsing
# ------------------------------


def parse_raster_date(
    name: str | Path,
    date_regex: str | re.Pattern | None = None,
) -> tuple[dt.date, str] | None:
    """
    Pull a date out of a raster file name.

    Recognizes ``YYYYMMDD``, ``YYYY-MM-DD``, ``YYYY_MM_DD`` and ``YYYY.MM.DD``
    anywhere in the name, and falls back to the year-month forms (``YYYYMM``,
    ``YYYY-MM``) used by the monthly footprint exports.  The first token that
    parses as a real calendar date wins.

    Parameters
    ----------
    name : str or pathlib.Path
        File name or full path.
    date_regex : str or re.Pattern, optional
        Override pattern.  Must expose either three groups (year, month, day)
        or two groups (year, month).

    Returns
    -------
    tuple of (datetime.date, str) or None
        The parsed date and its kind -- ``"day"`` or ``"month"`` (month-kind
        dates are anchored to the first of the month).  ``None`` if nothing in
        the name parses as a date.
    """
    stem = Path(name).name

    if date_regex is not None:
        pattern = re.compile(date_regex) if isinstance(date_regex, str) else date_regex
        for m in pattern.finditer(stem):
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 3:
                parsed = _safe_date(groups[0], groups[1], groups[2])
                if parsed is not None:
                    return parsed, "day"
            elif len(groups) == 2:
                parsed = _safe_date(groups[0], groups[1], "01")
                if parsed is not None:
                    return parsed, "month"
        return None

    for m in _DAY_RE.finditer(stem):
        parsed = _safe_date(*m.groups())
        if parsed is not None:
            return parsed, "day"

    for m in _MONTH_RE.finditer(stem):
        parsed = _safe_date(m.group(1), m.group(2), "01")
        if parsed is not None:
            return parsed, "month"

    return None


def _safe_date(year: str, month: str, day: str) -> dt.date | None:
    """Build a date, returning None for out-of-range or implausible tokens."""
    try:
        y, mo, d = int(year), int(month), int(day)
    except (TypeError, ValueError):
        return None
    if not (1900 <= y <= 2200):
        return None
    try:
        return dt.date(y, mo, d)
    except ValueError:
        return None


def _as_date(value: Any) -> dt.date:
    """Normalize a date-like key to ``datetime.date``."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def index_openet_rasters(
    source: OpenETSource,
    pattern: str = "*.tif",
    recursive: bool = False,
    date_regex: str | re.Pattern | None = None,
    logger: logging.Logger | None = None,
) -> dict[dt.date, list[Path]]:
    """
    Build a ``date -> [paths]`` index of daily OpenET rasters.

    Parameters
    ----------
    source : path, sequence of paths, or mapping
        A directory to scan, a single raster, a sequence of rasters, or an
        already-built mapping of date-like keys to one or more paths.
    pattern : str, default "*.tif"
        Glob used when `source` is a directory.
    recursive : bool, default False
        Search sub-directories as well.
    date_regex : str or re.Pattern, optional
        Custom date pattern passed to :func:`parse_raster_date`.
    logger : logging.Logger, optional

    Returns
    -------
    dict
        ``datetime.date`` keys mapped to sorted lists of paths.  Several files
        sharing a date (adjacent tiles, for instance) are kept together and
        combined when the mask is built.

    Notes
    -----
    Files whose name only carries a year and month are indexed under the first
    of that month, which still lets them serve a monthly footprint slice.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    index: dict[dt.date, list[Path]] = {}

    if isinstance(source, Mapping):
        for key, value in source.items():
            paths = (
                [Path(value)]
                if isinstance(value, (str, Path))
                else [Path(v) for v in value]
            )
            index.setdefault(_as_date(key), []).extend(paths)
    else:
        if isinstance(source, (str, Path)):
            root = Path(source)
            if root.is_dir():
                candidates = sorted(
                    root.rglob(pattern) if recursive else root.glob(pattern)
                )
            else:
                candidates = [root]
        elif isinstance(source, Iterable):
            candidates = [Path(p) for p in source]
        else:
            raise TypeError(
                f"Cannot interpret OpenET source of type {type(source)!r}. "
                "Pass a directory, a path, a sequence of paths, or a mapping."
            )

        for path in candidates:
            if not path.is_file():
                continue
            parsed = parse_raster_date(path, date_regex=date_regex)
            if parsed is None:
                logger.warning("No date found in OpenET file name: %s", path.name)
                continue
            date, kind = parsed
            if kind == "month":
                logger.warning(
                    "OpenET file %s carries only a year and month; "
                    "indexing it under %s.",
                    path.name,
                    date.isoformat(),
                )
            index.setdefault(date, []).append(path)

    index = {date: sorted(set(paths)) for date, paths in index.items()}

    if not index:
        raise ValueError(
            "No dated OpenET rasters found. Check the directory, the `pattern` "
            "glob, and that file names contain a YYYYMMDD-style date."
        )

    return dict(sorted(index.items()))


# ------------------------------
# Grid geometry
# ------------------------------


@dataclass(frozen=True)
class GridGeometry:
    """Georeferencing for a footprint grid expressed in metres from the tower."""

    crs: CRS
    transform: Affine
    width: int
    height: int
    x_origin: float
    y_origin: float
    y_ascending: bool

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width), i.e. numpy row/column order."""
        return (self.height, self.width)


def footprint_grid_geometry(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    station_lat: float,
    station_lon: float,
    crs: str | int | CRS = "auto",
) -> GridGeometry:
    """
    Georeference a footprint grid given in metres relative to the tower.

    Uses the same convention as :func:`~fluxfootprints.export_rasters_geotiff`:
    the tower is projected into the local WGS 84 UTM zone (or `crs`), grid
    offsets are added to it, and the resulting north-up transform is anchored on
    the outer edge of the corner cells.

    Parameters
    ----------
    x, y : array_like
        Regularly spaced cell-centre offsets in metres, as produced by the
        footprint models (``model.x``, ``model.y``).
    station_lat, station_lon : float
        Tower position in decimal degrees (WGS 84).
    crs : str, int, pyproj.CRS, default "auto"
        Target CRS; ``"auto"`` picks the local UTM zone.

    Returns
    -------
    GridGeometry
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("x/y grid must have at least 2 points each")

    xres = float(abs(x[1] - x[0]))
    yres = float(abs(y[1] - y[0]))

    if isinstance(crs, str) and crs == "auto":
        grid_crs = CRS.from_epsg(_choose_utm_epsg_pyproj(station_lon, station_lat))
    else:
        grid_crs = CRS.from_user_input(crs)

    to_proj = Transformer.from_crs(CRS.from_epsg(4326), grid_crs, always_xy=True)
    x0, y0 = to_proj.transform(station_lon, station_lat)

    left = (float(x.min()) + x0) - xres / 2.0
    top = (float(y.max()) + y0) + yres / 2.0

    return GridGeometry(
        crs=grid_crs,
        transform=from_origin(left, top, xres, yres),
        width=int(x.size),
        height=int(y.size),
        x_origin=float(x0),
        y_origin=float(y0),
        y_ascending=bool(y[1] > y[0]),
    )


# ------------------------------
# Mask construction
# ------------------------------


def _resolve_resampling(value: str | Resampling) -> Resampling:
    if isinstance(value, Resampling):
        return value
    try:
        return Resampling[str(value)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown resampling '{value}'. Use one of: "
            f"{[r.name for r in Resampling]}"
        ) from exc


def _resolve_band(src: rasterio.DatasetReader, band: int | str) -> int:
    """Map a band name or 1-based index onto a band index."""
    if isinstance(band, str):
        descriptions = [d or "" for d in src.descriptions]
        if band in descriptions:
            return descriptions.index(band) + 1
        raise KeyError(
            f"Band '{band}' not found in {Path(src.name).name}. "
            f"Available band descriptions: {descriptions}"
        )
    idx = int(band)
    if not 1 <= idx <= src.count:
        raise IndexError(
            f"Band {idx} out of range for {Path(src.name).name} "
            f"({src.count} band(s))"
        )
    return idx


def _native_valid_mask(
    path: str | Path,
    band: int | str = 1,
    nodata: float | None = None,
    valid_range: tuple[float | None, float | None] | None = None,
    treat_zero_as_nodata: bool = False,
) -> tuple[np.ndarray, CRS, Affine]:
    """
    Read one raster's valid-data mask in its own grid.

    A cell is valid when it is neither the dataset nodata value nor masked by an
    internal mask band, is finite, and -- when requested -- falls inside
    `valid_range` and is non-zero.
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                f"OpenET raster {Path(path).name} has no CRS; cannot align it "
                "to the footprint grid."
            )
        idx = _resolve_band(src, band)
        arr = src.read(idx, masked=True)
        src_crs = src.crs
        src_transform = src.transform

    valid = ~np.ma.getmaskarray(arr)
    data = np.asarray(arr.data, dtype="float64")

    valid &= np.isfinite(data)
    if nodata is not None:
        valid &= data != float(nodata)
    if treat_zero_as_nodata:
        valid &= data != 0.0
    if valid_range is not None:
        lo, hi = valid_range
        if lo is not None:
            valid &= data >= float(lo)
        if hi is not None:
            valid &= data <= float(hi)

    return valid, src_crs, src_transform


def _reproject_mask(
    valid: np.ndarray,
    src_crs: CRS,
    src_transform: Affine,
    dst_crs: CRS,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    resampling: Resampling,
    coverage_threshold: float,
) -> np.ndarray:
    """Warp a boolean mask onto the destination grid."""
    dst = np.zeros(dst_shape, dtype="float32")
    reproject(
        source=valid.astype("float32"),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0.0,
        resampling=resampling,
    )
    return dst >= float(coverage_threshold)


def _combine(masks: Sequence[np.ndarray], how: str) -> np.ndarray:
    """Reduce several boolean masks to one."""
    if not masks:
        raise ValueError("No masks to combine")
    stack = np.stack(masks)
    how = how.lower()
    if how == "any":
        return stack.any(axis=0)
    if how == "all":
        return stack.all(axis=0)
    if how == "majority":
        return stack.mean(axis=0) >= 0.5
    raise ValueError(
        f"Unknown mask combination '{how}'. Use 'any', 'all', or 'majority'."
    )


def openet_mask_on_grid(
    rasters: str | Path | Sequence[str | Path],
    dst_crs: CRS | str | int,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
    band: int | str = 1,
    nodata: float | None = None,
    valid_range: tuple[float | None, float | None] | None = None,
    treat_zero_as_nodata: bool = False,
    resampling: str | Resampling = "nearest",
    coverage_threshold: float = 0.5,
    combine: str = "any",
) -> np.ndarray:
    """
    Build a boolean valid-data mask for a target grid from OpenET rasters.

    Parameters
    ----------
    rasters : path or sequence of paths
        One or more OpenET rasters.  Several rasters are reduced with `combine`,
        which is how adjacent tiles for the same day are merged.
    dst_crs, dst_transform, dst_shape
        Target grid, north-up; `dst_shape` is ``(height, width)``.
    band : int or str, default 1
        Band index (1-based) or band description to read.
    nodata : float, optional
        Extra nodata value to honour on top of the dataset's own.
    valid_range : tuple of (float or None, float or None), optional
        Keep only cells inside these inclusive bounds, e.g. ``(0.0, None)`` to
        drop negative ET.
    treat_zero_as_nodata : bool, default False
        Treat exact zeros as missing.  Useful where OpenET writes 0 instead of
        nodata outside the crop mask; leave off when a real 0 mm/day matters.
    resampling : str or rasterio.warp.Resampling, default "nearest"
        Warping method.  ``"nearest"`` gives a hard mask; ``"average"`` gives
        the fraction of the destination cell covered by valid source cells,
        which is then thresholded with `coverage_threshold`.
    coverage_threshold : float, default 0.5
        Fraction of valid coverage a destination cell needs to count as valid.
    combine : {"any", "all", "majority"}, default "any"
        How to reduce multiple rasters.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape `dst_shape`; True where ET data exist.
    """
    paths = [rasters] if isinstance(rasters, (str, Path)) else list(rasters)
    if not paths:
        raise ValueError("No OpenET rasters given")

    target_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)
    method = _resolve_resampling(resampling)

    warped = []
    for path in paths:
        valid, src_crs, src_transform = _native_valid_mask(
            path,
            band=band,
            nodata=nodata,
            valid_range=valid_range,
            treat_zero_as_nodata=treat_zero_as_nodata,
        )
        warped.append(
            _reproject_mask(
                valid,
                src_crs,
                src_transform,
                target_crs,
                dst_transform,
                dst_shape,
                method,
                coverage_threshold,
            )
        )

    return _combine(warped, combine)


# ------------------------------
# Time-step selection
# ------------------------------


def _infer_freq(times: pd.DatetimeIndex) -> str:
    """Guess whether a footprint series is daily or monthly."""
    if len(times) > 1 and (times.day == 1).all():
        months = pd.PeriodIndex(times, freq="M")
        if months.is_unique:
            return "monthly"
    return "daily"


def _select_dates(
    ts: pd.Timestamp,
    freq: str,
    available: Sequence[dt.date],
    on_missing: str,
    max_gap_days: int,
) -> list[dt.date]:
    """Pick the OpenET dates that back one footprint time step."""
    if freq == "monthly":
        selected = [d for d in available if (d.year, d.month) == (ts.year, ts.month)]
    else:
        target = ts.date()
        selected = [d for d in available if d == target]

    if selected or on_missing != "nearest":
        return selected

    anchor = ts.date().replace(day=1) if freq == "monthly" else ts.date()
    nearest = min(available, key=lambda d: abs((d - anchor).days), default=None)
    if nearest is not None and abs((nearest - anchor).days) <= max_gap_days:
        return [nearest]
    return []


# ------------------------------
# Masking xarray footprints
# ------------------------------


@dataclass
class MaskedFootprint:
    """Result of masking a footprint field with OpenET valid-data masks."""

    data: xr.DataArray
    mask: xr.DataArray
    retained_fraction: xr.DataArray
    missing_dates: list[pd.Timestamp]


def mask_footprint_dataarray(
    da: xr.DataArray,
    openet: OpenETSource,
    station_lat: float,
    station_lon: float,
    freq: str = "auto",
    grid_crs: str | int | CRS = "auto",
    band: int | str = 1,
    nodata: float | None = None,
    valid_range: tuple[float | None, float | None] | None = None,
    treat_zero_as_nodata: bool = False,
    resampling: str | Resampling = "nearest",
    coverage_threshold: float = 0.5,
    combine: str = "any",
    on_missing: str = "skip",
    max_gap_days: int = 8,
    fill_value: float = 0.0,
    renormalize: bool = False,
    pattern: str = "*.tif",
    recursive: bool = False,
    date_regex: str | re.Pattern | None = None,
    logger: logging.Logger | None = None,
) -> MaskedFootprint:
    """
    Mask a time-resolved footprint field with daily OpenET valid-data masks.

    Each time step is matched to the OpenET raster(s) for that date; monthly
    steps (``freq="monthly"``) combine every OpenET day inside the month with
    `combine`.  Footprint weight on cells with no ET value is replaced by
    `fill_value`.

    Parameters
    ----------
    da : xarray.DataArray
        Footprint field with ``time``, ``x`` and ``y`` dims, where ``x``/``y``
        are metres from the tower -- e.g. ``model.f_2d`` or any field of
        :class:`~fluxfootprints.ffp_daily_monthly_helper.SummaryResult`.
    openet : path, sequence of paths, or mapping
        Daily OpenET rasters; see :func:`index_openet_rasters`.
    station_lat, station_lon : float
        Tower position in decimal degrees (WGS 84).
    freq : {"auto", "daily", "monthly"}, default "auto"
        How to interpret the time coordinate.  ``"auto"`` reads a series of more
        than one stamp, all first-of-month and one stamp per month, as monthly,
        and anything else as daily -- so set it explicitly for a single-slice
        monthly field.
    grid_crs : str, int, pyproj.CRS, default "auto"
        CRS the footprint grid is georeferenced into; ``"auto"`` uses the local
        UTM zone, matching the GeoTIFF export.
    band, nodata, valid_range, treat_zero_as_nodata, resampling,
    coverage_threshold, combine
        Passed through to :func:`openet_mask_on_grid`.
    on_missing : {"skip", "mask", "nearest", "error"}, default "skip"
        What to do for a time step with no OpenET raster: leave it unmasked,
        mask it out entirely, borrow the nearest date within `max_gap_days`, or
        raise.
    max_gap_days : int, default 8
        Search radius for ``on_missing="nearest"``.
    fill_value : float, default 0.0
        Value written where ET data are missing.  0.0 keeps the field usable by
        the contour and GeoTIFF exporters; pass ``numpy.nan`` to make the hole
        explicit.
    renormalize : bool, default False
        Rescale each masked time step so it integrates to the same total as
        before masking, i.e. redistribute the dropped weight over the cells that
        have ET.  Off by default so the loss stays visible in
        `retained_fraction`.
    pattern, recursive, date_regex
        Passed through to :func:`index_openet_rasters`.
    logger : logging.Logger, optional

    Returns
    -------
    MaskedFootprint
        ``data`` (masked field), ``mask`` (boolean, per time step), and
        ``retained_fraction`` (share of footprint weight surviving each step),
        plus ``missing_dates`` for steps with no OpenET coverage.

    Notes
    -----
    ``retained_fraction`` is the share of *this field's* weight kept, so it
    speaks to how much of the source area is observable by OpenET.  It is not
    the same as the domain-coverage columns produced by
    :func:`~fluxfootprints.summarize_periods`, which measure how much of the
    footprint fits inside the model domain.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    for dim in ("time", "x", "y"):
        if dim not in da.dims:
            raise ValueError(
                f"DataArray must have a '{dim}' dimension; got {da.dims}. "
                "Pass a time-resolved footprint field."
            )

    on_missing = on_missing.lower()
    if on_missing not in {"skip", "mask", "nearest", "error"}:
        raise ValueError(
            f"Unknown on_missing '{on_missing}'. "
            "Use 'skip', 'mask', 'nearest', or 'error'."
        )

    index = index_openet_rasters(
        openet,
        pattern=pattern,
        recursive=recursive,
        date_regex=date_regex,
        logger=logger,
    )
    available = list(index.keys())

    times = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    freq = freq.lower()
    if freq == "auto":
        freq = _infer_freq(times)
    elif freq not in {"daily", "monthly"}:
        raise ValueError(f"Unknown freq '{freq}'. Use 'auto', 'daily', or 'monthly'.")

    geom = footprint_grid_geometry(
        da["x"].values, da["y"].values, station_lat, station_lon, crs=grid_crs
    )

    cache: dict[tuple[Path, ...], np.ndarray] = {}
    layers: list[np.ndarray] = []
    missing: list[pd.Timestamp] = []

    for ts in times:
        dates = _select_dates(ts, freq, available, on_missing, max_gap_days)

        if not dates:
            missing.append(ts)
            if on_missing == "error":
                raise FileNotFoundError(
                    f"No OpenET raster for {ts.date().isoformat()} "
                    f"({freq} footprint slice)."
                )
            logger.warning(
                "No OpenET raster for %s; %s.",
                ts.date().isoformat(),
                (
                    "masking the slice out"
                    if on_missing == "mask"
                    else "leaving it unmasked"
                ),
            )
            fill = on_missing != "mask"
            layers.append(np.full(geom.shape, fill, dtype=bool))
            continue

        paths = tuple(p for d in dates for p in index[d])
        if paths not in cache:
            cache[paths] = openet_mask_on_grid(
                paths,
                geom.crs,
                geom.transform,
                geom.shape,
                band=band,
                nodata=nodata,
                valid_range=valid_range,
                treat_zero_as_nodata=treat_zero_as_nodata,
                resampling=resampling,
                coverage_threshold=coverage_threshold,
                combine=combine,
            )
        layers.append(cache[paths])

    stack = np.stack(layers)  # (time, y, x), north-up
    if geom.y_ascending:
        # The grid geometry is north-up (first row = highest y) while this
        # DataArray's y coordinate ascends, so flip rows to line them up.
        stack = stack[:, ::-1, :]

    mask = xr.DataArray(
        stack,
        dims=("time", "y", "x"),
        coords={"time": da["time"], "y": da["y"], "x": da["x"]},
        name="openet_valid",
    )

    original_total = da.fillna(0.0).sum(dim=("x", "y"))
    # Weight kept is measured on the footprint itself, not on the filled field,
    # so a non-zero fill_value cannot inflate the retained fraction.
    kept_total = da.where(mask, 0.0).fillna(0.0).sum(dim=("x", "y"))
    retained = kept_total / original_total.where(original_total != 0)

    masked = da.where(mask, fill_value).transpose(*da.dims)

    if renormalize:
        scale = original_total / kept_total.where(kept_total != 0)
        masked = xr.where(mask, masked * scale, masked).transpose(*da.dims)

    masked.attrs = dict(da.attrs)
    masked.attrs.update(
        {
            "openet_masked": "true",
            "openet_freq": freq,
            "openet_combine": combine,
            "openet_on_missing": on_missing,
            "openet_fill_value": fill_value,
            "openet_renormalized": str(bool(renormalize)).lower(),
            "openet_n_dates": len(index),
            "openet_missing_slices": len(missing),
        }
    )
    masked.name = da.name

    if missing:
        logger.warning(
            "%d of %d %s footprint slices had no OpenET raster.",
            len(missing),
            len(times),
            freq,
        )

    return MaskedFootprint(
        data=masked,
        mask=mask,
        retained_fraction=retained.rename("openet_retained_frac"),
        missing_dates=missing,
    )


def mask_summaries(
    summaries: SummaryResult,
    openet: OpenETSource,
    station_lat: float,
    station_lon: float,
    **kwargs: Any,
) -> SummaryResult:
    """
    Apply OpenET masks to every field of a :class:`SummaryResult`.

    Daily fields are matched day-by-day and monthly fields month-by-month.  The
    surviving fraction of footprint weight is appended to the coverage tables as
    ``openet_retained_frac`` (and ``openet_retained_frac_etw`` for the
    ET-weighted fields).

    Parameters
    ----------
    summaries : SummaryResult
        Output of :func:`~fluxfootprints.summarize_periods`.
    openet : path, sequence of paths, or mapping
        Daily OpenET rasters; see :func:`index_openet_rasters`.
    station_lat, station_lon : float
        Tower position in decimal degrees (WGS 84).
    **kwargs
        Any keyword accepted by :func:`mask_footprint_dataarray` except `freq`,
        which is set per field.

    Returns
    -------
    SummaryResult
        New object; the input is not modified.
    """
    kwargs.pop("freq", None)

    fields = {
        "f_daily_mean": "daily",
        "f_daily_et_weighted": "daily",
        "f_monthly_mean": "monthly",
        "f_monthly_et_weighted": "monthly",
    }

    out = SummaryResult()
    retained: dict[str, xr.DataArray] = {}

    for name, freq in fields.items():
        da = getattr(summaries, name, None)
        if da is None:
            continue
        result = mask_footprint_dataarray(
            da,
            openet,
            station_lat=station_lat,
            station_lon=station_lon,
            freq=freq,
            **kwargs,
        )
        setattr(out, name, result.data)
        retained[name] = result.retained_fraction

    for attr, mean_key, etw_key in (
        ("daily_domain_coverage", "f_daily_mean", "f_daily_et_weighted"),
        ("monthly_domain_coverage", "f_monthly_mean", "f_monthly_et_weighted"),
    ):
        table = getattr(summaries, attr, None)
        if table is None:
            continue
        table = table.copy()
        if mean_key in retained:
            table["openet_retained_frac"] = retained[mean_key].to_series()
        if etw_key in retained:
            table["openet_retained_frac_etw"] = retained[etw_key].to_series()
        setattr(out, attr, table)

    return out


# ------------------------------
# Masking exported GeoTIFFs
# ------------------------------


def mask_rasters_geotiff(
    raster_dir: str | Path,
    openet: OpenETSource,
    out_dir: str | Path | None = None,
    pattern: str = "*.tif",
    fill_value: float | None = None,
    band: int | str = 1,
    nodata: float | None = None,
    valid_range: tuple[float | None, float | None] | None = None,
    treat_zero_as_nodata: bool = False,
    resampling: str | Resampling = "nearest",
    coverage_threshold: float = 0.5,
    combine: str = "any",
    on_missing: str = "skip",
    max_gap_days: int = 8,
    renormalize: bool = False,
    openet_pattern: str = "*.tif",
    recursive: bool = False,
    date_regex: str | re.Pattern | None = None,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Mask footprint GeoTIFFs on disk with daily OpenET valid-data masks.

    Each footprint raster is matched to OpenET by the date in its file name --
    ``..._20240201.tif`` uses that day, ``..._202402.tif`` combines every OpenET
    day in February 2024 -- so this works directly on the output of
    :func:`~fluxfootprints.export_rasters_geotiff`.  Masks are built against
    each file's own CRS and transform, so reprojected exports are handled too.

    Parameters
    ----------
    raster_dir : path
        Directory of footprint GeoTIFFs (or a single GeoTIFF).
    openet : path, sequence of paths, or mapping
        Daily OpenET rasters; see :func:`index_openet_rasters`.
    out_dir : path, optional
        Where masked copies are written.  Defaults to
        ``<raster_dir>/openet_masked``.  Inputs are never overwritten in place
        unless `out_dir` is set to the input directory.
    pattern : str, default "*.tif"
        Glob for the footprint rasters.
    fill_value : float, optional
        Value written where ET data are missing.  Defaults to each file's own
        nodata value, or 0.0 when it has none.
    band, nodata, valid_range, treat_zero_as_nodata, resampling,
    coverage_threshold, combine
        Passed through to :func:`openet_mask_on_grid` (they describe the OpenET
        rasters, not the footprint rasters).
    on_missing : {"skip", "mask", "nearest", "error"}, default "skip"
        What to do when no OpenET raster matches a footprint raster's date:
        copy it through unmasked, blank it, borrow the nearest date within
        `max_gap_days`, or raise.
    max_gap_days : int, default 8
        Search radius for ``on_missing="nearest"``.
    renormalize : bool, default False
        Rescale each masked band to its pre-masking sum.
    openet_pattern, recursive, date_regex
        Passed through to :func:`index_openet_rasters`.
    logger : logging.Logger, optional

    Returns
    -------
    list of pathlib.Path
        The masked rasters that were written, in name order.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    on_missing = on_missing.lower()
    if on_missing not in {"skip", "mask", "nearest", "error"}:
        raise ValueError(
            f"Unknown on_missing '{on_missing}'. "
            "Use 'skip', 'mask', 'nearest', or 'error'."
        )

    raster_dir = Path(raster_dir)
    if raster_dir.is_dir():
        sources = sorted(p for p in raster_dir.glob(pattern) if p.is_file())
        default_out = raster_dir / "openet_masked"
    elif raster_dir.is_file():
        sources = [raster_dir]
        default_out = raster_dir.parent / "openet_masked"
    else:
        raise FileNotFoundError(f"No such file or directory: {raster_dir}")

    if not sources:
        raise ValueError(f"No rasters matching '{pattern}' in {raster_dir}")

    out_dir = Path(out_dir) if out_dir is not None else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    index = index_openet_rasters(
        openet,
        pattern=openet_pattern,
        recursive=recursive,
        date_regex=date_regex,
        logger=logger,
    )
    available = list(index.keys())

    written: list[Path] = []

    for source in sources:
        parsed = parse_raster_date(source)
        if parsed is None:
            logger.warning("No date in raster name %s; skipping.", source.name)
            continue
        date, kind = parsed
        freq = "monthly" if kind == "month" else "daily"

        dates = _select_dates(
            pd.Timestamp(date), freq, available, on_missing, max_gap_days
        )
        if not dates:
            if on_missing == "error":
                raise FileNotFoundError(
                    f"No OpenET raster for {date.isoformat()} (from {source.name})."
                )
            if on_missing == "skip":
                logger.warning(
                    "No OpenET raster for %s; copying %s through unmasked.",
                    date.isoformat(),
                    source.name,
                )

        with rasterio.open(source) as src:
            profile = src.profile.copy()
            data = src.read().astype("float64")
            src_nodata = src.nodata

        if dates:
            paths = tuple(p for d in dates for p in index[d])
            mask = openet_mask_on_grid(
                paths,
                profile["crs"],
                profile["transform"],
                (profile["height"], profile["width"]),
                band=band,
                nodata=nodata,
                valid_range=valid_range,
                treat_zero_as_nodata=treat_zero_as_nodata,
                resampling=resampling,
                coverage_threshold=coverage_threshold,
                combine=combine,
            )
        elif on_missing == "mask":
            mask = np.zeros((profile["height"], profile["width"]), dtype=bool)
        else:
            mask = np.ones((profile["height"], profile["width"]), dtype=bool)

        fill = fill_value
        if fill is None:
            fill = float(src_nodata) if src_nodata is not None else 0.0

        out_data = np.where(mask[None, :, :], data, fill)

        if renormalize:
            # Rescale on real values only, so the file's own nodata cells
            # neither contribute to the sum nor get scaled.
            data_valid = np.isfinite(data)
            if src_nodata is not None and np.isfinite(src_nodata):
                data_valid &= data != src_nodata
            keep = mask[None, :, :] & data_valid
            before = np.where(data_valid, data, 0.0).sum(axis=(1, 2))
            after = np.where(keep, data, 0.0).sum(axis=(1, 2))
            scale = np.divide(before, after, out=np.ones_like(before), where=after > 0)
            out_data = np.where(keep, data * scale[:, None, None], out_data)

        dest = out_dir / source.name
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(out_data.astype(profile["dtype"]))
        written.append(dest)

    return sorted(written)


# ------------------------------
# Dispatcher
# ------------------------------


def apply_openet_mask(
    target: xr.DataArray | SummaryResult | str | Path,
    openet: OpenETSource,
    station_lat: float | None = None,
    station_lon: float | None = None,
    **kwargs: Any,
) -> MaskedFootprint | SummaryResult | list[Path]:
    """
    Mask footprint output with daily OpenET valid-data masks.

    Convenience entry point that dispatches on `target`:

    ==============================  ==========================================
    `target`                        behaviour
    ==============================  ==========================================
    :class:`xarray.DataArray`       :func:`mask_footprint_dataarray`
    :class:`SummaryResult`          :func:`mask_summaries`
    path to a directory or GeoTIFF  :func:`mask_rasters_geotiff`
    ==============================  ==========================================

    Parameters
    ----------
    target : xarray.DataArray, SummaryResult, or path
        What to mask.
    openet : path, sequence of paths, or mapping
        Daily OpenET rasters; see :func:`index_openet_rasters`.
    station_lat, station_lon : float, optional
        Tower position in decimal degrees (WGS 84).  Required for the xarray
        targets, which carry grid offsets rather than map coordinates; ignored
        for GeoTIFFs, which are already georeferenced.
    **kwargs
        Forwarded to the dispatched function.

    Returns
    -------
    MaskedFootprint, SummaryResult, or list of pathlib.Path
        Matching the dispatched function.
    """
    if isinstance(target, (xr.DataArray, SummaryResult)):
        if station_lat is None or station_lon is None:
            raise ValueError(
                "station_lat and station_lon are required to georeference a "
                "footprint grid given in metres from the tower."
            )
        if isinstance(target, xr.DataArray):
            return mask_footprint_dataarray(
                target, openet, station_lat, station_lon, **kwargs
            )
        return mask_summaries(target, openet, station_lat, station_lon, **kwargs)

    if isinstance(target, (str, Path)):
        return mask_rasters_geotiff(target, openet, **kwargs)

    raise TypeError(
        f"Cannot mask object of type {type(target)!r}. Pass a DataArray, a "
        "SummaryResult, or a path to footprint GeoTIFFs."
    )
