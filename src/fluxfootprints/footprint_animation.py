"""Animate footprint raster time series as video.

This module turns the time-resolved footprint rasters produced by the models in
:mod:`fluxfootprints` (or the daily/monthly summaries from
:func:`fluxfootprints.summarize_periods`) into a video or animated GIF.  Each
frame is one time step, annotated with its timestamp, and the raster can be
drawn on top of a georeferenced web-map basemap.

The main entry points are :class:`FootprintAnimator` (full control, reusable)
and :func:`animate_footprint` (one-shot convenience wrapper).

Examples
--------
Animate the native (half-hourly) footprint time series aggregated to hours::

    from fluxfootprints import animate_footprint

    animate_footprint(model, "ffp_hourly.gif", freq="hourly")

Animate the daily-mean summary on an aerial basemap::

    animate_footprint(
        summaries.f_daily_mean,
        "ffp_daily.mp4",
        freq=None,                 # already daily
        station_lat=40.05,
        station_lon=-113.55,
        basemap=True,
    )

Notes
-----
Writing ``.mp4``/``.webm`` needs an ``ffmpeg`` binary and basemaps need
``contextily``; both come with the optional extra::

    pip install "fluxfootprints[animation]"

``ffmpeg`` already on the system ``PATH`` is used in preference to the copy
bundled by ``imageio-ffmpeg``.  ``.gif`` output needs neither -- it only uses
Pillow, which matplotlib already depends on.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from pyproj import CRS, Transformer

from .ffp_daily_monthly_helper import _choose_utm_epsg_pyproj

__all__ = [
    "FootprintAnimator",
    "animate_footprint",
    "ensure_ffmpeg",
    "resolve_freq",
    "resample_footprints",
]

logger = logging.getLogger(__name__)

# ------------------------------
# Frequency handling
# ------------------------------

#: Human-friendly frequency names mapped to pandas offset aliases.
_FREQ_ALIASES: dict[str, str | None] = {
    "hourly": "1h",
    "hour": "1h",
    "h": "1h",
    "1h": "1h",
    "daily": "1D",
    "day": "1D",
    "d": "1D",
    "1d": "1D",
    "monthly": "MS",
    "month": "MS",
    "ms": "MS",
    "native": None,
    "raw": None,
    "none": None,
}

#: Default timestamp label format for each supported frequency.
_DEFAULT_TIME_FORMATS: dict[str | None, str] = {
    "1h": "%Y-%m-%d %H:%M",
    "1D": "%Y-%m-%d",
    "MS": "%B %Y",
    None: "%Y-%m-%d %H:%M",
}

TimeStep = Literal["hourly", "daily", "monthly", "native"]


def resolve_freq(freq: str | None) -> str | None:
    """Translate a friendly time-step name to a pandas offset alias.

    Parameters
    ----------
    freq : str or None
        ``"hourly"``, ``"daily"``, ``"monthly"``, ``"native"``/``None``, or any
        pandas offset alias (e.g. ``"3h"``, ``"7D"``) which is passed through.

    Returns
    -------
    str or None
        The pandas offset alias, or ``None`` to keep the native time steps.

    Raises
    ------
    ValueError
        If ``freq`` is not a string or ``None``.
    """
    if freq is None:
        return None
    if not isinstance(freq, str):
        raise ValueError(f"freq must be a string or None, got {type(freq)!r}")

    key = freq.strip().lower()
    if key in _FREQ_ALIASES:
        return _FREQ_ALIASES[key]
    # Unrecognised names are assumed to be pandas offset aliases ("3h", "7D", ...)
    return freq


def resample_footprints(
    da: xr.DataArray,
    freq: str | None = None,
    reducer: str = "mean",
    normalize_each_frame: bool = False,
) -> xr.DataArray:
    """Aggregate a footprint time series onto a coarser time step.

    Parameters
    ----------
    da : xarray.DataArray
        Footprint densities with a ``time`` dimension plus ``x`` and ``y``.
    freq : str or None, optional
        Target time step; see :func:`resolve_freq`.  ``None`` keeps the native
        time steps.
    reducer : {"mean", "sum", "median", "max"}, optional
        Reduction applied within each period.  ``"mean"`` is appropriate for
        normalized footprint densities.
    normalize_each_frame : bool, optional
        If ``True``, rescale each resulting frame so it sums to one.  Useful
        when comparing footprint *shape* rather than magnitude.

    Returns
    -------
    xarray.DataArray
        The aggregated series, with all-NaN periods dropped.
    """
    offset = resolve_freq(freq)
    out = da

    if offset is not None:
        grouped = out.resample(time=offset)
        try:
            out = getattr(grouped, reducer)(skipna=True)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown reducer '{reducer}'. Use one of: mean, sum, median, max."
            ) from exc

    # Drop periods with no data at all (gaps introduced by resampling).
    other_dims = [d for d in out.dims if d != "time"]
    finite = out.notnull().any(dim=other_dims)
    out = out.isel(time=np.flatnonzero(np.asarray(finite.values)))

    if normalize_each_frame:
        totals = out.sum(dim=("x", "y"), skipna=True)
        out = out / totals.where(totals > 0)

    return out


# ------------------------------
# Input coercion helpers
# ------------------------------


def _as_dataarray(data: Any, var: str | None = None) -> xr.DataArray:
    """Coerce a model, dataset, or raw array into a 3-D footprint DataArray."""
    if isinstance(data, xr.DataArray):
        da = data
    elif isinstance(data, xr.Dataset):
        if var is None:
            candidates = [
                name
                for name, v in data.data_vars.items()
                if "time" in v.dims and {"x", "y"} <= set(v.dims)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Dataset has {len(candidates)} time-resolved 2-D variables "
                    f"{candidates}; pass var= to choose one."
                )
            var = candidates[0]
        da = data[var]
    elif hasattr(data, "get_footprint_timeseries"):
        da = data.get_footprint_timeseries()
        if da is None:
            raise ValueError(
                f"{type(data).__name__} does not expose a time-resolved footprint "
                "(get_footprint_timeseries() returned None)."
            )
    else:
        raise TypeError(
            "data must be an xarray.DataArray, xarray.Dataset, or a footprint "
            f"model exposing get_footprint_timeseries(); got {type(data)!r}"
        )

    missing = {"time", "x", "y"} - set(da.dims)
    if missing:
        raise ValueError(
            f"Footprint array is missing required dimension(s) {sorted(missing)}; "
            f"found {list(da.dims)}"
        )
    return da


def _cell_size(da: xr.DataArray) -> tuple[float, float]:
    """Return the ``(dx, dy)`` grid spacing in metres."""
    x = np.asarray(da["x"].values, dtype=float)
    y = np.asarray(da["y"].values, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("x/y grids must each have at least two points")
    return float(abs(x[1] - x[0])), float(abs(y[1] - y[0]))


def _source_area_levels(
    field: np.ndarray, cell_area: float, fractions: Sequence[float]
) -> list[float]:
    """Density values bounding the requested source-area fractions.

    Sorts the field in descending order and walks the cumulative integral until
    each requested fraction of the total contribution is enclosed, mirroring the
    contour convention used elsewhere in the package.
    """
    flat = field[np.isfinite(field)]
    if flat.size == 0:
        return []

    sf = np.sort(flat)[::-1]
    csf = np.cumsum(sf) * cell_area

    levels: list[float] = []
    for frac in fractions:
        idx = int(np.argmin(np.abs(csf - frac)))
        levels.append(float(sf[idx]))

    # contour() needs strictly increasing, unique levels
    return sorted(set(levels))


def ensure_ffmpeg() -> str | None:
    """Locate an ffmpeg binary for matplotlib's video writers.

    Prefers whatever matplotlib is already configured to use (normally
    ``ffmpeg`` on the system ``PATH``) and otherwise falls back to the binary
    shipped with the optional ``imageio-ffmpeg`` package, pointing matplotlib
    at it so :class:`~matplotlib.animation.FFMpegWriter` becomes usable.

    Returns
    -------
    str or None
        Path to a working ffmpeg executable, or ``None`` if none was found.
    """
    from matplotlib import animation as mplanim

    if mplanim.writers.is_available("ffmpeg"):
        return str(matplotlib.rcParams["animation.ffmpeg_path"])

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.debug("imageio-ffmpeg could not provide a binary: %s", exc)
        return None

    matplotlib.rcParams["animation.ffmpeg_path"] = exe
    if not mplanim.writers.is_available("ffmpeg"):
        return None
    return str(exe)


_TEXT_POSITIONS: dict[str, tuple[float, float, str, str]] = {
    "upper left": (0.025, 0.975, "top", "left"),
    "upper right": (0.975, 0.975, "top", "right"),
    "lower left": (0.025, 0.025, "bottom", "left"),
    "lower right": (0.975, 0.025, "bottom", "right"),
    "upper center": (0.5, 0.975, "top", "center"),
    "lower center": (0.5, 0.025, "bottom", "center"),
}


# ------------------------------
# Animator
# ------------------------------


class FootprintAnimator:
    """Build an animation from a time-resolved footprint raster.

    The animator resolves the input to a ``(time, y, x)`` raster stack,
    optionally aggregates it to hourly/daily/monthly steps, places it in a
    projected CRS when station coordinates are supplied, and renders one frame
    per time step with a timestamp label.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, or footprint model
        Source of the footprint time series.  Models are read through
        ``get_footprint_timeseries()``; datasets need ``var`` if they hold more
        than one time-resolved 2-D variable.
    var : str, optional
        Variable name when ``data`` is a :class:`xarray.Dataset`.
    freq : {"hourly", "daily", "monthly", "native"} or str or None, optional
        Time step of the output animation.  ``None``/``"native"`` keeps the
        input time steps; any pandas offset alias also works.
    reducer : str, optional
        Aggregation applied when resampling; see :func:`resample_footprints`.
    normalize_each_frame : bool, optional
        Rescale every frame to sum to one before plotting.
    time_slice : tuple of str, optional
        ``(start, stop)`` labels used to subset the time axis before animating.
    station_lat, station_lon : float, optional
        Tower position in WGS84.  Both are required to georeference the raster
        (and therefore to draw a basemap); without them the axes are plotted in
        metres relative to the tower.
    crs_out : str or int, optional
        Target CRS for georeferenced output.  ``"auto"`` picks the local UTM
        zone.  Must be a projected CRS.
    basemap : bool, optional
        Draw a web-map basemap under the raster.  Requires ``contextily`` and
        station coordinates.
    basemap_source : optional
        A ``contextily`` tile provider (defaults to the ``contextily`` default).
    basemap_zoom : int or str, optional
        Tile zoom level passed to ``contextily.add_basemap``.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the footprint density.
    norm : {"linear", "log"}, optional
        Color scaling of the density values.
    color_scale : {"global", "per_frame"}, optional
        Whether the color limits are fixed across the whole animation
        (comparable frames) or recomputed per frame (maximum contrast).
    vmin, vmax : float, optional
        Explicit color limits; override ``percentiles``.
    percentiles : tuple of float, optional
        Percentiles of the finite, positive data used to derive color limits.
    log_decades : float, optional
        Dynamic range of the log color scale, in decades below ``vmax``.  Only
        used when ``norm="log"`` and ``vmin`` is not given: footprint fields
        trail off to arbitrarily small densities, so the lowest percentile is a
        useless floor for a log scale.
    alpha : float, optional
        Raster opacity.  Defaults to 0.75 when a basemap is drawn, else 1.0.
    mask_below : float, optional
        Densities at or below this absolute value are drawn transparent.
    mask_quantile : float, optional
        Same as ``mask_below`` but expressed as a quantile (0-1) of the finite,
        positive densities.
    contour_fractions : sequence of float, optional
        Source-area fractions (e.g. ``(0.5, 0.8)``) contoured on each frame.
    contour_color, contour_width : optional
        Styling for those contours.
    title : str, optional
        Static figure title drawn above the axes.
    timestamp_format : str, optional
        ``strftime`` pattern for the per-frame label; defaults follow ``freq``.
    timestamp_loc : str or tuple, optional
        One of the ``"upper left"``-style keys, or an ``(x, y)`` pair in axes
        fractions.
    timestamp_kwargs : dict, optional
        Extra keyword arguments forwarded to :meth:`matplotlib.axes.Axes.text`.
    annotation_fn : callable, optional
        ``f(timestamp, frame_array) -> str`` returning a second line of text
        appended under the timestamp.
    show_colorbar : bool, optional
        Draw a colorbar for the density.
    cbar_label : str, optional
        Colorbar label.
    show_tower : bool, optional
        Mark the tower position.
    figsize : tuple of float, optional
        Figure size in inches.
    dpi : int, optional
        Figure and output resolution.
    fps : int, optional
        Frames per second of the saved animation.
    logger : logging.Logger, optional
        Logger used for progress messages.

    Attributes
    ----------
    frames : xarray.DataArray
        The prepared ``(time, y, x)`` stack that will be animated.
    times : pandas.DatetimeIndex
        Timestamps of the frames.
    extent : tuple of float
        ``(left, right, bottom, top)`` of the raster in plot coordinates.
    crs : pyproj.CRS or None
        CRS of the plot coordinates, or ``None`` when plotting in local metres.
    """

    def __init__(
        self,
        data: Any,
        var: str | None = None,
        freq: str | None = None,
        reducer: str = "mean",
        normalize_each_frame: bool = False,
        time_slice: tuple[str, str] | None = None,
        station_lat: float | None = None,
        station_lon: float | None = None,
        crs_out: str | int = "auto",
        basemap: bool = False,
        basemap_source: Any = None,
        basemap_zoom: int | str = "auto",
        cmap: str | Colormap = "viridis",
        norm: Literal["linear", "log"] = "linear",
        color_scale: Literal["global", "per_frame"] = "global",
        vmin: float | None = None,
        vmax: float | None = None,
        percentiles: tuple[float, float] = (0.0, 99.5),
        log_decades: float = 4.0,
        alpha: float | None = None,
        mask_below: float | None = None,
        mask_quantile: float | None = None,
        contour_fractions: Sequence[float] | None = None,
        contour_color: str = "white",
        contour_width: float = 0.8,
        title: str | None = None,
        timestamp_format: str | None = None,
        timestamp_loc: str | tuple[float, float] = "upper left",
        timestamp_kwargs: dict[str, Any] | None = None,
        annotation_fn: Callable[[pd.Timestamp, np.ndarray], str] | None = None,
        show_colorbar: bool = True,
        cbar_label: str = "Flux contribution (m$^{-2}$)",
        show_tower: bool = True,
        figsize: tuple[float, float] = (8.0, 7.0),
        dpi: int = 120,
        fps: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.log = logger or logging.getLogger(__name__)
        self.freq = resolve_freq(freq)
        self.fps = int(fps)
        self.dpi = int(dpi)
        self.figsize = figsize

        # --- 1. Resolve and shape the raster stack --------------------------
        da = _as_dataarray(data, var=var)
        if time_slice is not None:
            da = da.sel(time=slice(*time_slice))
        da = resample_footprints(
            da,
            freq=self.freq,
            reducer=reducer,
            normalize_each_frame=normalize_each_frame,
        )
        if da.sizes["time"] == 0:
            raise ValueError("No frames left to animate after resampling/slicing.")

        # imshow(origin="lower") wants (row=y ascending, col=x ascending)
        da = da.transpose("time", "y", "x").sortby("x").sortby("y")
        self.frames = da
        self.times = pd.to_datetime(da["time"].values)
        self.dx, self.dy = _cell_size(da)
        self.cell_area = self.dx * self.dy

        # --- 2. Geolocate the grid ------------------------------------------
        self.x0 = 0.0
        self.y0 = 0.0
        self.crs, self.extent = self._build_extent(station_lat, station_lon, crs_out)
        self.station_lat = station_lat
        self.station_lon = station_lon

        # --- 3. Rendering options -------------------------------------------
        self.basemap = bool(basemap)
        if self.basemap and self.crs is None:
            raise ValueError(
                "basemap=True requires station_lat and station_lon so the "
                "footprint grid can be georeferenced."
            )
        self.basemap_source = basemap_source
        self.basemap_zoom = basemap_zoom

        base_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        # Masked/NaN cells stay fully transparent so a basemap shows through.
        self.cmap = base_cmap.with_extremes(bad=(0, 0, 0, 0))
        self.norm_kind = norm
        self.log_decades = float(log_decades)
        self.color_scale = color_scale
        self.alpha = alpha if alpha is not None else (0.75 if self.basemap else 1.0)
        self.contour_fractions = list(contour_fractions) if contour_fractions else []
        self.contour_color = contour_color
        self.contour_width = contour_width

        self.title = title
        self.timestamp_format = timestamp_format or _DEFAULT_TIME_FORMATS.get(
            self.freq, "%Y-%m-%d %H:%M"
        )
        self.timestamp_loc = timestamp_loc
        self.timestamp_kwargs = timestamp_kwargs or {}
        self.annotation_fn = annotation_fn
        self.show_colorbar = show_colorbar
        self.cbar_label = cbar_label
        self.show_tower = show_tower

        # --- 4. Color limits and masking -------------------------------------
        self.mask_threshold = self._resolve_mask(mask_below, mask_quantile)
        self.vmin, self.vmax = self._resolve_color_limits(vmin, vmax, percentiles)

        self.log.info(
            "Prepared %d frames (%s) spanning %s to %s",
            len(self.times),
            self.freq or "native",
            self.times[0],
            self.times[-1],
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_extent(
        self,
        station_lat: float | None,
        station_lon: float | None,
        crs_out: str | int,
    ) -> tuple[CRS | None, tuple[float, float, float, float]]:
        """Return the plot CRS and pixel-edge extent of the raster."""
        x = np.asarray(self.frames["x"].values, dtype=float)
        y = np.asarray(self.frames["y"].values, dtype=float)

        left = x.min() - self.dx / 2.0
        right = x.max() + self.dx / 2.0
        bottom = y.min() - self.dy / 2.0
        top = y.max() + self.dy / 2.0

        if station_lat is None or station_lon is None:
            if station_lat is not None or station_lon is not None:
                raise ValueError("Provide both station_lat and station_lon, or neither.")
            return None, (left, right, bottom, top)

        if isinstance(crs_out, str) and crs_out == "auto":
            target = CRS.from_epsg(_choose_utm_epsg_pyproj(station_lon, station_lat))
        else:
            target = CRS.from_user_input(crs_out)

        if not target.is_projected:
            raise ValueError(
                f"crs_out must be a projected CRS with metre units; "
                f"{target.to_string()} is geographic. Use 'auto' for local UTM."
            )

        to_proj = Transformer.from_crs(CRS.from_epsg(4326), target, always_xy=True)
        self.x0, self.y0 = to_proj.transform(station_lon, station_lat)
        return target, (
            left + self.x0,
            right + self.x0,
            bottom + self.y0,
            top + self.y0,
        )

    def _resolve_mask(
        self, mask_below: float | None, mask_quantile: float | None
    ) -> float | None:
        """Resolve the transparency threshold to an absolute density."""
        if mask_below is not None and mask_quantile is not None:
            raise ValueError("Pass mask_below or mask_quantile, not both.")
        if mask_below is not None:
            return float(mask_below)
        if mask_quantile is None:
            return None
        if not 0.0 <= mask_quantile < 1.0:
            raise ValueError("mask_quantile must be in [0, 1)")

        positive = self.frames.where(self.frames > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = float(positive.quantile(mask_quantile, skipna=True))
        return value if np.isfinite(value) else None

    def _resolve_color_limits(
        self,
        vmin: float | None,
        vmax: float | None,
        percentiles: tuple[float, float],
    ) -> tuple[float, float]:
        """Derive color limits from the finite, positive densities."""
        positive = self.frames.where(np.isfinite(self.frames) & (self.frames > 0))
        lo_p, hi_p = percentiles
        with warnings.catch_warnings():
            # An all-zero footprint leaves nothing to take quantiles of; the
            # NaN result is handled just below.
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = float(positive.quantile(lo_p / 100.0, skipna=True))
            hi = float(positive.quantile(hi_p / 100.0, skipna=True))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0:
            # Degenerate (all-zero / all-NaN) input: fall back to a unit range.
            lo, hi = 0.0, 1.0

        vmin_given = vmin is not None
        vmin = float(vmin) if vmin_given else lo
        vmax = float(vmax) if vmax is not None else hi

        if self.norm_kind == "log":
            # Footprint densities trail off towards zero, so the low percentile
            # is a meaningless floor on a log scale (it lands many decades below
            # anything visible). Clamp the range to log_decades instead.
            floor = vmax / 10.0**self.log_decades if vmax > 0 else 1e-12
            if not vmin_given:
                vmin = max(vmin, floor)
            if vmin <= 0:
                vmin = floor
        if vmax <= vmin:
            vmax = vmin * 10.0 if self.norm_kind == "log" else vmin + 1.0
        return vmin, vmax

    def _make_norm(self, vmin: float, vmax: float) -> Normalize:
        """Build the color normalization for the chosen scaling."""
        if self.norm_kind == "log":
            return LogNorm(vmin=max(vmin, np.finfo(float).tiny), vmax=vmax)
        if self.norm_kind != "linear":
            raise ValueError(f"norm must be 'linear' or 'log', got {self.norm_kind!r}")
        return Normalize(vmin=vmin, vmax=vmax)

    def _frame_array(self, index: int) -> np.ndarray:
        """Return one frame as a float array with masked cells set to NaN."""
        arr = np.asarray(self.frames.isel(time=index).values, dtype=float)
        if self.mask_threshold is not None:
            arr = np.where(arr > self.mask_threshold, arr, np.nan)
        if self.norm_kind == "log":
            arr = np.where(arr > 0, arr, np.nan)
        return arr

    def _add_basemap(self, ax: plt.Axes) -> None:
        """Draw a contextily basemap behind the raster (no-op if disabled)."""
        if not self.basemap:
            return
        try:
            import contextily as cx
        except ImportError as exc:
            raise ImportError(
                "basemap=True requires the optional 'contextily' package. "
                "Install it with: pip install contextily"
            ) from exc

        kwargs: dict[str, Any] = {"crs": self.crs, "attribution_size": 6}
        if self.basemap_source is not None:
            kwargs["source"] = self.basemap_source
        if self.basemap_zoom != "auto":
            kwargs["zoom"] = self.basemap_zoom

        try:
            cx.add_basemap(ax, **kwargs)
        except Exception as exc:
            # Tile servers fail often enough that losing the basemap should not
            # cost the caller the whole animation.
            self.log.warning(
                "Basemap could not be fetched (%s); continuing without it.", exc
            )

    def _timestamp_xy(self) -> tuple[float, float, str, str]:
        """Resolve the timestamp anchor to ``(x, y, va, ha)`` in axes fractions."""
        loc = self.timestamp_loc
        if isinstance(loc, str):
            if loc not in _TEXT_POSITIONS:
                raise ValueError(
                    f"timestamp_loc must be one of {sorted(_TEXT_POSITIONS)} "
                    "or an (x, y) pair in axes fractions."
                )
            return _TEXT_POSITIONS[loc]
        x, y = loc
        return float(x), float(y), "top", "left"

    def _label(self, index: int, arr: np.ndarray) -> str:
        """Compose the text drawn on a frame."""
        text = self.times[index].strftime(self.timestamp_format)
        if self.annotation_fn is not None:
            extra = self.annotation_fn(self.times[index], arr)
            if extra:
                text = f"{text}\n{extra}"
        return text

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def build(self) -> tuple[Figure, FuncAnimation]:
        """Create the figure and its :class:`~matplotlib.animation.FuncAnimation`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure being animated.  Close it when finished.
        anim : matplotlib.animation.FuncAnimation
            The animation, ready to save or embed (``anim.to_jshtml()``).
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        first = self._frame_array(0)
        norm = self._make_norm(self.vmin, self.vmax)

        # Fix the axes to the raster extent before the basemap is fetched, so
        # contextily requests tiles for the right window.
        left, right, bottom, top = self.extent
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        self._add_basemap(ax)

        im = ax.imshow(
            first,
            extent=self.extent,
            origin="lower",
            cmap=self.cmap,
            norm=norm,
            alpha=self.alpha,
            interpolation="nearest",
            zorder=2,
        )
        # imshow rescales the view; restore the basemap window.
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)

        if self.crs is not None:
            ax.set_xlabel(f"Easting [m] ({self.crs.to_string()})")
            ax.set_ylabel("Northing [m]")
        else:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

        if self.show_tower:
            ax.plot(
                self.x0,
                self.y0,
                marker="^",
                color="black",
                markersize=9,
                markeredgecolor="white",
                linestyle="none",
                zorder=4,
                label="Tower",
            )

        if self.title:
            ax.set_title(self.title)

        if self.show_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.8, format="%.2e", label=self.cbar_label)

        tx, ty, va, ha = self._timestamp_xy()
        text_kwargs: dict[str, Any] = {
            "fontsize": 12,
            "color": "black",
            "bbox": {
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "none",
                "pad": 4,
            },
        }
        text_kwargs.update(self.timestamp_kwargs)
        label = ax.text(
            tx,
            ty,
            self._label(0, first),
            transform=ax.transAxes,
            va=va,
            ha=ha,
            zorder=5,
            **text_kwargs,
        )

        # Contours are re-created each frame; track the current set so it can be
        # removed before the next one is drawn.
        state: dict[str, Any] = {"contours": None}
        x_edges = np.linspace(left, right, self.frames.sizes["x"])
        y_edges = np.linspace(bottom, top, self.frames.sizes["y"])

        def _draw_contours(arr: np.ndarray) -> None:
            if state["contours"] is not None:
                state["contours"].remove()
                state["contours"] = None
            if not self.contour_fractions:
                return
            levels = _source_area_levels(arr, self.cell_area, self.contour_fractions)
            if not levels:
                return
            state["contours"] = ax.contour(
                x_edges,
                y_edges,
                np.nan_to_num(arr, nan=0.0),
                levels=levels,
                colors=self.contour_color,
                linewidths=self.contour_width,
                zorder=3,
            )

        _draw_contours(first)

        def update(index: int):
            arr = self._frame_array(index)
            im.set_data(arr)
            if self.color_scale == "per_frame":
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    hi = float(finite.max())
                    lo = self.vmin if self.norm_kind == "log" else 0.0
                    if hi > lo:
                        im.set_clim(lo, hi)
            label.set_text(self._label(index, arr))
            _draw_contours(arr)
            return (im, label)

        # Frames are rendered at a fixed size, so lay the axes out now rather
        # than relying on bbox_inches="tight" (which video writers cannot use).
        fig.tight_layout()

        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.times),
            interval=1000.0 / max(self.fps, 1),
            blit=False,
            repeat=False,
        )
        # Keep the per-frame renderer reachable for _save_frames().
        self._update_frame = update
        return fig, anim

    def save(
        self,
        path: str | Path,
        writer: str | Any | None = None,
        frames_dir: str | Path | None = None,
        **writer_kwargs: Any,
    ) -> Path:
        """Render the animation to a video or GIF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file.  The suffix selects the writer: ``.mp4``/``.mov``/
            ``.m4v``/``.webm`` use ffmpeg, ``.gif`` uses Pillow, ``.html``
            writes an embedded HTML player.
        writer : str or matplotlib writer, optional
            Override the writer chosen from the suffix.
        frames_dir : str or pathlib.Path, optional
            If given, also write each frame as a numbered PNG into this
            directory.
        **writer_kwargs
            Extra keyword arguments passed to the writer (e.g.
            ``bitrate=2400``, ``codec="libx264"``).

        Returns
        -------
        pathlib.Path
            The path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if writer is None:
            writer = self._writer_for(path.suffix.lower(), **writer_kwargs)

        fig, anim = self.build()
        total = len(self.times)
        try:
            self.log.info("Writing %d frames to %s", total, path)
            anim.save(
                str(path),
                writer=writer,
                dpi=self.dpi,
                progress_callback=lambda i, n: self.log.debug("frame %d/%d", i + 1, n),
            )
            if frames_dir is not None:
                self._save_frames(fig, frames_dir)
        finally:
            plt.close(fig)

        self.log.info("Wrote %s", path)
        return path

    def _writer_for(self, suffix: str, **writer_kwargs: Any) -> Any:
        """Pick a matplotlib writer for the requested output suffix."""
        from matplotlib import animation as mplanim

        video_codecs = {".mp4": "h264", ".m4v": "h264", ".mov": "h264", ".webm": "vp9"}

        if suffix == ".gif":
            return mplanim.PillowWriter(fps=self.fps, **writer_kwargs)
        if suffix in (".html", ".htm"):
            return mplanim.HTMLWriter(fps=self.fps, **writer_kwargs)
        if suffix in video_codecs:
            if ensure_ffmpeg() is None:
                raise RuntimeError(
                    f"Writing '{suffix}' needs an ffmpeg binary, which was not "
                    'found. Install it with: pip install "fluxfootprints[animation]" '
                    "(or put ffmpeg on your PATH), or save to '.gif' instead."
                )
            writer_kwargs.setdefault("codec", video_codecs[suffix])
            return mplanim.FFMpegWriter(fps=self.fps, **writer_kwargs)

        raise ValueError(
            f"Unsupported output extension '{suffix}'. Use .mp4, .mov, .m4v, "
            ".webm, .gif, or .html."
        )

    def _save_frames(self, fig: Figure, frames_dir: str | Path) -> None:
        """Dump every frame as a numbered PNG alongside the animation."""
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        width = len(str(len(self.times)))
        for i, ts in enumerate(self.times):
            self._update_frame(i)
            stamp = pd.Timestamp(ts).strftime("%Y%m%dT%H%M")
            fig.savefig(
                frames_dir / f"frame_{i:0{width}d}_{stamp}.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_summary(
        cls,
        summaries: Any,
        layer: str = "daily_mean",
        **kwargs: Any,
    ) -> "FootprintAnimator":
        """Build an animator from a ``SummaryResult``.

        Parameters
        ----------
        summaries : SummaryResult
            Output of :func:`fluxfootprints.summarize_periods`.
        layer : {"daily_mean", "monthly_mean", "daily_etw", "monthly_etw"}
            Which summary raster to animate.  The timestamp format follows the
            layer's frequency unless overridden in ``kwargs``.
        **kwargs
            Forwarded to :class:`FootprintAnimator`.

        Returns
        -------
        FootprintAnimator
        """
        layer_map = {
            "daily_mean": ("f_daily_mean", "daily"),
            "monthly_mean": ("f_monthly_mean", "monthly"),
            "daily_etw": ("f_daily_et_weighted", "daily"),
            "monthly_etw": ("f_monthly_et_weighted", "monthly"),
        }
        if layer not in layer_map:
            raise ValueError(f"Unknown layer '{layer}'. Choose from: {sorted(layer_map)}")
        attr, freq = layer_map[layer]
        da = getattr(summaries, attr, None)
        if da is None:
            raise ValueError(
                f"Summary layer '{layer}' is empty; re-run summarize_periods with "
                "the matching options enabled."
            )

        # The summary is already aggregated, so the layer's frequency only sets
        # the label format unless the caller overrides it.
        kwargs.setdefault("timestamp_format", _DEFAULT_TIME_FORMATS[resolve_freq(freq)])
        kwargs.setdefault("freq", None)
        return cls(da, **kwargs)


def animate_footprint(
    data: Any,
    path: str | Path,
    freq: str | None = None,
    **kwargs: Any,
) -> Path:
    """Render a footprint time series to a video or GIF in one call.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, or footprint model
        Source of the time-resolved footprint raster.
    path : str or pathlib.Path
        Output file; the suffix selects the format (``.mp4``, ``.gif``, ...).
    freq : {"hourly", "daily", "monthly", "native"} or None, optional
        Time step of the animation.
    **kwargs
        Any other :class:`FootprintAnimator` argument, plus ``writer``,
        ``frames_dir``, and writer options which are forwarded to
        :meth:`FootprintAnimator.save`.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Examples
    --------
    >>> animate_footprint(model, "daily.gif", freq="daily")  # doctest: +SKIP
    PosixPath('daily.gif')
    """
    save_keys = {"writer", "frames_dir", "bitrate", "codec", "metadata", "extra_args"}
    save_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in save_keys}

    animator = FootprintAnimator(data, freq=freq, **kwargs)
    return animator.save(path, **save_kwargs)
