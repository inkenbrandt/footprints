# src/fluxfootprints/representativeness_plotting.py
"""
representativeness_plotting.py
==============================
Figures for the footprint-to-target-area representativeness analysis.

Draws the four diagnostics of :mod:`~fluxfootprints.representativeness` in the
form Chu et al. (2021) publish them:

    Chu, H., et al. (2021). Representativeness of Eddy-Covariance flux
    footprints for areas surrounding AmeriFlux sites. *Agricultural and Forest
    Meteorology*, **301-302**, 108350.
    https://doi.org/10.1016/j.agrformet.2021.108350

===================================== ==========================
Function                              Figure in the paper
===================================== ==========================
:func:`plot_landcover_composition`    Fig. 1e
:func:`plot_footprint_target_scatter` Fig. 1f, Fig. 6
:func:`plot_bias_density`             Fig. 7
:func:`plot_level_bars`               Fig. 5, Fig. 8
===================================== ==========================

Each takes what the analysis functions already return -- a list of
:class:`~fluxfootprints.CategoricalResult`, or the tidy frames of
:func:`~fluxfootprints.sensor_location_bias_series` and
:func:`~fluxfootprints.assess_representativeness` -- and each returns
``(fig, ax)`` without showing or saving anything, so the caller keeps control
of the title, the layout, and the output.

Colour convention
-----------------
Target-area radius is an *ordered* quantity, so it is drawn on a single-hue
sequential ramp (:data:`RADIUS_CMAP`), dark at the smallest radius and light at
the largest -- as the paper's own legends read, "from dark to light, indicating
an increasing distance from the tower". The three-level index gets its own
ordered ramp (:data:`LEVEL_CMAP`), dark for HIGH through light for LOW. The
footprint-weighted series is not a step of either sequence: it is the reference
the target areas are read against, so it wears a single contrasting accent
(:data:`FOOTPRINT_COLOR`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from .representativeness import CategoricalResult, Level, rma_regression

__all__ = [
    "RADIUS_CMAP",
    "LEVEL_CMAP",
    "FOOTPRINT_COLOR",
    "REFERENCE_COLOR",
    "radius_colors",
    "level_colors",
    "plot_landcover_composition",
    "plot_footprint_target_scatter",
    "plot_bias_density",
    "plot_level_bars",
]


# ------------------------------
# Style constants
# ------------------------------

#: Sequential colormap for series keyed by target-area radius, sampled
#: dark-to-light with increasing radius.
RADIUS_CMAP: str = "Oranges"

#: Sequential colormap for the three-level representativeness index, sampled
#: dark for HIGH through light for LOW.
LEVEL_CMAP: str = "Blues"

#: Accent for the footprint-weighted series, which is the reference the target
#: areas are read against rather than one more step of the radius ramp.
FOOTPRINT_COLOR: str = "#1b4f9c"

#: Ink for reference geometry -- the 1:1 line, the zero-bias rule.
REFERENCE_COLOR: str = "#767676"

#: Ink for the hairline grid, one shade off the surface.
GRID_COLOR: str = "#dcdcdc"

#: Fraction of a colormap the ramps are sampled over, darkest first. Stopping
#: short of both ends keeps the lightest step visible on white and the darkest
#: step from reading as black.
_RAMP_BAND: tuple[float, float] = (0.85, 0.38)

#: Order the three-level index is drawn in, darkest first.
_LEVEL_ORDER: tuple[Level, ...] = (Level.HIGH, Level.MEDIUM, Level.LOW)

#: Columns searched for the per-period sensor location bias, in order.
#: ``"delta"`` is what :func:`sensor_location_bias_series` names it, ``"bias"``
#: what :func:`assess_representativeness` names it.
_BIAS_COLUMNS: tuple[str, ...] = ("delta", "bias")

#: Columns searched for a three-level index, in order.
_LEVEL_COLUMNS: tuple[str, ...] = ("level", "landcover_level", "continuous_level")


# ------------------------------
# Shared helpers
# ------------------------------


def _sample_ramp(name: str, n: int) -> list[tuple[float, float, float, float]]:
    """
    Sample `n` steps from a sequential colormap, darkest first.

    Parameters
    ----------
    name : str
        Name of a Matplotlib colormap, e.g. ``"Oranges"``.
    n : int
        Number of steps. A single step takes the dark end of the band.

    Returns
    -------
    list of tuple
        RGBA tuples, dark to light.
    """
    cmap = colormaps[name]
    if n <= 1:
        return [cmap(_RAMP_BAND[0])]
    return [cmap(position) for position in np.linspace(*_RAMP_BAND, n)]


def radius_colors(
    radii: Sequence[float],
    cmap: str = RADIUS_CMAP,
) -> dict[float, tuple[float, float, float, float]]:
    """
    Map target-area radii onto the sequential ramp, dark to light.

    The step a radius gets depends on its rank within `radii` rather than on
    its value, so the ramp spans the series actually drawn and two figures over
    the same series of radii agree.

    Parameters
    ----------
    radii : sequence of float
        Target-area radii [m]. Sorted ascending here; duplicates collapse.
    cmap : str, default RADIUS_CMAP
        Name of a Matplotlib sequential colormap.

    Returns
    -------
    dict
        Radius -> RGBA, the smallest radius darkest.

    See Also
    --------
    level_colors : The same for the three-level index.

    Examples
    --------
    >>> sorted(radius_colors([3000, 250]))
    [250.0, 3000.0]
    """
    unique = sorted({float(radius) for radius in radii})
    return dict(zip(unique, _sample_ramp(cmap, len(unique))))


def level_colors(
    cmap: str = LEVEL_CMAP,
) -> dict[Level, tuple[float, float, float, float]]:
    """
    Map the three-level index onto its ramp, HIGH darkest.

    Parameters
    ----------
    cmap : str, default LEVEL_CMAP
        Name of a Matplotlib sequential colormap.

    Returns
    -------
    dict
        :class:`~fluxfootprints.Level` -> RGBA.

    See Also
    --------
    radius_colors : The same for target-area radii.

    Examples
    --------
    >>> list(level_colors())
    [<Level.HIGH: 'high'>, <Level.MEDIUM: 'medium'>, <Level.LOW: 'low'>]
    """
    return dict(zip(_LEVEL_ORDER, _sample_ramp(cmap, len(_LEVEL_ORDER))))


def _axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """
    Resolve the axes to draw on, opening a figure when none is given.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw on, or None to open a new figure.
    figsize : tuple of float
        Size of the new figure [in], ignored when `ax` is given.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _style(ax: Axes, axis: str = "both") -> None:
    """
    Apply recessive chrome to an axes, in place.

    A solid hairline grid one shade off the surface, and no top or right
    spine, so that the marks rather than the frame carry the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style.
    axis : {'both', 'x', 'y'}, default 'both'
        Which grid to draw.
    """
    ax.grid(True, axis=axis, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#8c8c8c")


def _numeric(values: Any) -> np.ndarray:
    """
    Coerce a column to float64, missing where it does not convert.

    Handles the nullable dtypes the analysis frames carry, which
    ``astype(float)`` refuses once a value is missing.

    Parameters
    ----------
    values : array_like
        The column, Series, or array to coerce.

    Returns
    -------
    numpy.ndarray
        Float64, ``nan`` where the input was missing or unparseable.
    """
    return pd.to_numeric(pd.Series(np.asarray(values, dtype=object)), errors="coerce").to_numpy(
        dtype=float
    )


def _as_frame(data: pd.DataFrame, required: Sequence[str], name: str) -> pd.DataFrame:
    """
    Normalise a results frame so the columns a plot needs are columns.

    The tidy frame of :func:`~fluxfootprints.assess_representativeness` carries
    ``radius`` and the rest of :data:`~fluxfootprints.RESULT_INDEX` in its
    index, while :func:`~fluxfootprints.sensor_location_bias_series` carries
    them as columns. Either is accepted here by resetting any index level a
    required name is hiding in.

    Parameters
    ----------
    data : pandas.DataFrame
        The frame to normalise.
    required : sequence of str
        Names the caller needs as columns. Absent ones are reported together.
    name : str
        Name of the argument, for error messages.

    Returns
    -------
    pandas.DataFrame
        A frame carrying every name in `required` as a column: a reset copy
        when a level had to be lifted out of the index, otherwise the input.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame.
    ValueError
        If a required name is neither a column nor an index level.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, not {type(data).__name__}. "
            "Pass the frame from sensor_location_bias_series, "
            "assess_representativeness, or representativeness_summary."
        )

    levels = [level for level in data.index.names if level is not None]
    hidden = [
        column for column in required if column not in data.columns and column in levels
    ]
    frame = data.reset_index() if hidden else data

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing the column(s) {missing}; it carries "
            f"{list(frame.columns)}. Name the columns to read with the "
            "*_column arguments if they go by other names here."
        )
    return frame


def _selected_radii(
    values: Any,
    radii: Sequence[float] | None,
    name: str,
) -> list[float]:
    """
    Resolve which radii to draw, ascending.

    Parameters
    ----------
    values : array_like
        Radii present in the data, coerced to float here.
    radii : sequence of float or None
        The subset the caller asked for, or None for every radius present.
    name : str
        Name of the data argument, for error messages.

    Returns
    -------
    list of float
        The radii to draw, ascending.

    Raises
    ------
    ValueError
        If the data holds no finite radius, or if a requested radius is absent.
    """
    numeric = _numeric(values)
    present = sorted({float(value) for value in numeric if np.isfinite(value)})
    if not present:
        raise ValueError(f"{name} holds no finite target-area radius to group by.")
    if radii is None:
        return present

    wanted = sorted({float(radius) for radius in radii})
    absent = [radius for radius in wanted if radius not in present]
    if absent:
        raise ValueError(f"radii {absent} are not in {name}, which holds {present}.")
    return wanted


def _detect_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    explicit: str | None,
    name: str,
    role: str,
) -> str:
    """
    Resolve a column the caller named, or find it by convention.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame to search.
    candidates : sequence of str
        Conventional names, in search order.
    explicit : str or None
        The name the caller gave, which wins outright.
    name : str
        Name of the data argument, for error messages.
    role : str
        What the column holds, for error messages.

    Returns
    -------
    str
        The resolved column name.

    Raises
    ------
    ValueError
        If `explicit` is absent from `frame`, or if no candidate is present.
    """
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(
                f"{name} has no column {explicit!r}; it carries "
                f"{list(frame.columns)}."
            )
        return explicit

    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(
        f"{name} carries none of the {role} columns {list(candidates)}; it holds "
        f"{list(frame.columns)}. Name the column to draw explicitly."
    )


def _radius_label(radius: float) -> str:
    """
    Render a radius as a legend entry.

    Parameters
    ----------
    radius : float
        Target-area radius [m].

    Returns
    -------
    str
        e.g. ``"250 m"``.
    """
    return f"{radius:g} m"


# ------------------------------
# Fig. 1e -- land-cover composition
# ------------------------------


def plot_landcover_composition(
    results: Sequence[CategoricalResult],
    *,
    class_labels: Mapping[Any, str] | None = None,
    max_classes: int | None = None,
    radii: Sequence[float] | None = None,
    ax: Axes | None = None,
    cmap: str = RADIUS_CMAP,
    footprint_color: str = FOOTPRINT_COLOR,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> tuple[Figure, Axes]:
    """
    Plot footprint-weighted against target-area land-cover shares, Fig. 1e.

    One row per land-cover class, ordered by footprint-weighted share. The
    footprint-weighted percentage is a filled circle in the accent colour, and
    each target area a triangle on the radius ramp, so the drift of a class's
    share as the disc widens reads as a trail away from the circle.

    Parameters
    ----------
    results : sequence of CategoricalResult
        The output of :func:`~fluxfootprints.evaluate_landcover`, one entry per
        radius. Every entry carries the same footprint composition -- the
        footprint does not depend on the disc -- so the first entry's is drawn.
    class_labels : mapping, optional
        Class code -> display name, e.g. ``{41: "Deciduous forest"}``. A code
        with no entry is labelled by the code itself.
    max_classes : int, optional
        Keep only this many classes, those with the largest share in either the
        footprint or any target area. Every class present is drawn when omitted.
    radii : sequence of float, optional
        Draw only these radii. The ramp is still sampled over every radius in
        `results`, so a subset keeps the colours of the full series.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is opened when omitted.
    cmap : str, default RADIUS_CMAP
        Sequential colormap for the target-area series.
    footprint_color : str, default FOOTPRINT_COLOR
        Colour of the footprint-weighted series.
    figsize : tuple of float, default (7.0, 4.5)
        Size of a new figure [in]. Ignored when `ax` is given.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and the axes drawn on. Nothing is shown or saved.

    Raises
    ------
    ValueError
        If `results` is empty, if `radii` names a radius `results` does not
        hold, if `max_classes` is not positive, or if no class carries a share
        in either the footprint or a target area.

    See Also
    --------
    fluxfootprints.evaluate_landcover : Produces `results`.
    plot_level_bars : The same evaluation reduced to its three-level index and
        pooled across sites.

    Notes
    -----
    Shares are the percentages :class:`~fluxfootprints.CategoricalResult`
    carries, so a column of circles sums to 100 % only when `max_classes`
    dropped nothing.

    Examples
    --------
    >>> results = evaluate_landcover(model.fclim_2d, nlcd)   # doctest: +SKIP
    >>> fig, ax = plot_landcover_composition(                # doctest: +SKIP
    ...     results, class_labels={41: "Deciduous forest", 82: "Cropland"}
    ... )
    >>> fig.savefig("landcover.png", dpi=200)                # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Fig. 1e.
    """
    entries = list(results)
    if not entries:
        raise ValueError(
            "results is empty, so there is no composition to draw. Pass the "
            "output of evaluate_landcover."
        )
    if max_classes is not None and max_classes < 1:
        raise ValueError(f"max_classes must be positive, got {max_classes}.")

    available = [float(entry.radius) for entry in entries]
    keep = _selected_radii(available, radii, "results")
    colors = radius_colors(available, cmap=cmap)

    footprint = dict(entries[0].footprint_composition)
    targets = {
        float(entry.radius): dict(entry.target_composition)
        for entry in entries
        if float(entry.radius) in keep
    }

    def _largest_target(code: Any) -> float:
        return max(float(share.get(code, 0.0)) for share in targets.values())

    codes = set(footprint) | {code for share in targets.values() for code in share}
    ranked = sorted(
        (
            code
            for code in codes
            if max(float(footprint.get(code, 0.0)), _largest_target(code)) > 0.0
        ),
        key=lambda code: (-float(footprint.get(code, 0.0)), -_largest_target(code)),
    )
    if not ranked:
        raise ValueError(
            "No land-cover class carries a share in either the footprint or a "
            "target area, so there is nothing to draw."
        )
    if max_classes is not None:
        ranked = ranked[:max_classes]

    fig, ax = _axes(ax, figsize)
    positions = np.arange(len(ranked), dtype=float)

    for radius in keep:
        share = targets[radius]
        ax.plot(
            [float(share.get(code, 0.0)) for code in ranked],
            positions,
            marker="v",
            markersize=6.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="none",
            color=colors[radius],
            label=_radius_label(radius),
            zorder=2,
        )

    ax.plot(
        [float(footprint.get(code, 0.0)) for code in ranked],
        positions,
        marker="o",
        markersize=7.0,
        markeredgecolor="white",
        markeredgewidth=0.6,
        linestyle="none",
        color=footprint_color,
        label="Footprint-weighted",
        zorder=3,
    )

    labels = class_labels or {}
    ax.set_yticks(positions)
    ax.set_yticklabels([str(labels.get(code, code)) for code in ranked])
    ax.invert_yaxis()
    ax.set_xlim(-2.0, 102.0)
    ax.set_xlabel("Land-cover percentage (%)")
    _style(ax, axis="x")

    # The footprint is the reference, so it leads the legend rather than
    # trailing the radii it was drawn over.
    handles, texts = ax.get_legend_handles_labels()
    order = [len(handles) - 1, *range(len(handles) - 1)]
    ax.legend(
        [handles[index] for index in order],
        [texts[index] for index in order],
        loc="best",
        frameon=False,
        fontsize="small",
        handletextpad=0.4,
        borderaxespad=0.6,
    )
    return fig, ax


# ------------------------------
# Fig. 1f, Fig. 6 -- footprint against target area
# ------------------------------


def plot_footprint_target_scatter(
    data: pd.DataFrame,
    *,
    radii: Sequence[float] | None = None,
    x_column: str = "value_footprint",
    y_column: str = "value_target",
    radius_column: str | None = "radius",
    fit: bool = True,
    annotate: bool | None = None,
    variable: str = "EVI",
    ax: Axes | None = None,
    cmap: str = RADIUS_CMAP,
    figsize: tuple[float, float] = (5.5, 5.5),
) -> tuple[Figure, Axes]:
    """
    Plot target-area against footprint-weighted values, Fig. 1f and Fig. 6.

    Scatters the matched pairs of Eq. 5, one series per target-area radius,
    under the reduced major axis fit of Eq. 7 and the 1:1 line. Points settling
    below 1:1 are the paper's central result: the footprint saw a higher value
    than the disc drawn around it.

    Parameters
    ----------
    data : pandas.DataFrame
        Matched values, one row per period and radius: either the frame of
        :func:`~fluxfootprints.sensor_location_bias_series` or the tidy frame
        of :func:`~fluxfootprints.assess_representativeness`, whose ``radius``
        index level is reset here. Subset the latter to one ``variable`` and
        ``period`` first -- every row given is drawn.
    radii : sequence of float, optional
        Draw only these radii. Every radius present is drawn when omitted.
    x_column, y_column : str
        Columns holding the footprint-weighted and target-area values,
        defaulting to the names both frames use.
    radius_column : str or None, default 'radius'
        Column keying the series. Pass None to draw the frame as a single
        series, the one-panel form of Fig. 6.
    fit : bool, default True
        Draw the RMA line of :func:`~fluxfootprints.rma_regression` through
        each series. A series of fewer than three finite pairs is scattered
        without one.
    annotate : bool, optional
        Write the fit statistics into the corner. Defaults to True for a single
        series and False for several, where the box would crowd the panel and
        the legend already carries the identities.
    variable : str, default 'EVI'
        Name of the field, used in the axis labels.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is opened when omitted.
    cmap : str, default RADIUS_CMAP
        Sequential colormap for the radius series.
    figsize : tuple of float, default (5.5, 5.5)
        Size of a new figure [in]. Ignored when `ax` is given.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and the axes drawn on, at equal aspect and with one range on
        both axes, so that the 1:1 line runs corner to corner.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame.
    ValueError
        If a named column is neither a column nor an index level of `data`, if
        `radii` names an absent radius, or if no finite pair survives.

    See Also
    --------
    fluxfootprints.rma_regression : The model II fit drawn here.
    fluxfootprints.evaluate_vegetation_index : Reduces the same pairs to a
        three-level index.
    plot_bias_density : The same mismatch as a distribution of Eq. 6.

    Notes
    -----
    The line is the reduced major axis rather than ordinary least squares:
    both axes are spatial averages of one noisy raster and both carry error, so
    an OLS slope would sit systematically shallower and the slopes of Table 1
    are reproducible only against the RMA. Each line is drawn across the range
    of its own series, never extrapolated past it.

    Examples
    --------
    >>> bias = sensor_location_bias_series(pairs, model.x, model.y)  # doctest: +SKIP
    >>> fig, ax = plot_footprint_target_scatter(bias)                # doctest: +SKIP
    >>> fig, ax = plot_footprint_target_scatter(                     # doctest: +SKIP
    ...     bias[bias["radius"] == 250], radius_column=None
    ... )

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350,
    Fig. 1f, Fig. 6, Table 1.
    """
    required = [x_column, y_column]
    if radius_column is not None:
        required.append(radius_column)
    frame = _as_frame(data, required, "data")

    if radius_column is None:
        groups: list[tuple[float | None, pd.DataFrame]] = [(None, frame)]
        colors: dict[Any, Any] = {None: FOOTPRINT_COLOR}
    else:
        column = _numeric(frame[radius_column])
        keep = _selected_radii(column, radii, "data")
        ramp = radius_colors(column[np.isfinite(column)], cmap=cmap)
        groups = [(radius, frame[column == radius]) for radius in keep]
        colors = {radius: ramp[radius] for radius in keep}

    fig, ax = _axes(ax, figsize)
    finite: list[np.ndarray] = []
    fits: list[tuple[float | None, Any]] = []

    for key, group in groups:
        xs = _numeric(group[x_column])
        ys = _numeric(group[y_column])
        good = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[good], ys[good]
        if xs.size == 0:
            continue
        finite.extend((xs, ys))

        color = colors[key]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.4,
            linestyle="none",
            alpha=0.75,
            color=color,
            # A lone series is named by the axis labels; only the several need
            # telling apart.
            label=_radius_label(key) if key is not None else None,
            zorder=2,
        )

        if fit and xs.size >= 3:
            result = rma_regression(xs, ys)
            if np.isfinite(result.slope) and np.isfinite(result.intercept):
                span = np.array([xs.min(), xs.max()], dtype=float)
                ax.plot(
                    span,
                    result.intercept + result.slope * span,
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=3,
                )
                fits.append((key, result))

    if not finite:
        raise ValueError(
            f"No finite ({x_column}, {y_column}) pair survives in data, so "
            "there is nothing to scatter."
        )

    pooled = np.concatenate(finite)
    low, high = float(pooled.min()), float(pooled.max())
    pad = 0.05 * (high - low) if high > low else 0.5
    limits = (low - pad, high + pad)
    ax.plot(
        limits,
        limits,
        linestyle="--",
        linewidth=1.0,
        color=REFERENCE_COLOR,
        label="1:1",
        zorder=1,
    )

    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Footprint-weighted {variable}")
    ax.set_ylabel(f"Target-area {variable}")
    _style(ax)

    if annotate is None:
        annotate = len(groups) == 1
    if annotate and fits:
        key, result = fits[0]
        heading = "" if key is None else f"{_radius_label(key)}\n"
        ax.text(
            0.04,
            0.96,
            f"{heading}slope = {result.slope:.2f}\n"
            f"intercept = {result.intercept:.2f}\n"
            f"$R^2$ = {result.r_squared:.2f}\n"
            f"n = {result.n}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize="small",
            color="#333333",
        )

    ax.legend(loc="lower right", frameon=False, fontsize="small", handletextpad=0.4)
    return fig, ax


# ------------------------------
# Fig. 7 -- sensor location bias
# ------------------------------


def plot_bias_density(
    data: pd.DataFrame,
    *,
    radii: Sequence[float] | None = None,
    bias_column: str | None = None,
    radius_column: str = "radius",
    percent: bool = True,
    clip: tuple[float, float] = (-100.0, 100.0),
    bandwidth: float | str | None = None,
    gridsize: int = 512,
    threshold: float | None = None,
    ax: Axes | None = None,
    cmap: str = RADIUS_CMAP,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> tuple[Figure, Axes]:
    """
    Plot kernel densities of the sensor location bias by radius, Fig. 7.

    One Gaussian kernel density per target area over the per-period biases of
    Eq. 6, drawn on the radius ramp. The peak sharpens and rises towards the
    smallest radius, which is the paper's Fig. 7 result: the wider the disc,
    the further a month's footprint-weighted value strays from it.

    Parameters
    ----------
    data : pandas.DataFrame
        Per-period biases, either the frame of
        :func:`~fluxfootprints.sensor_location_bias_series` (column ``delta``)
        or the tidy frame of :func:`~fluxfootprints.assess_representativeness`
        (column ``bias``), whose ``radius`` index level is reset here. Subset
        the latter to one ``variable`` and ``period`` first.
    radii : sequence of float, optional
        Draw only these radii. Every radius present is drawn when omitted.
    bias_column : str, optional
        Column holding the bias as a *fraction*. Detected among ``"delta"`` and
        ``"bias"`` when omitted.
    radius_column : str, default 'radius'
        Column keying the densities.
    percent : bool, default True
        Draw the bias as a percentage, as the paper's axis does. Set False to
        keep the fractions the analysis returns.
    clip : tuple of float, default (-100.0, 100.0)
        Range the densities are drawn over, in the units of the axis. Every
        finite bias enters its kernel, including those outside this window;
        only the drawing is clipped, as the paper's +/-100 % axis is.
    bandwidth : float or str, optional
        Passed to :class:`scipy.stats.gaussian_kde` as ``bw_method``. Scott's
        rule when omitted.
    gridsize : int, default 512
        Number of points each density is evaluated on.
    threshold : float, optional
        Draw guides at ``+/- threshold``, as a fraction -- pass
        :data:`~fluxfootprints.BIAS_THRESHOLD` for the +/-10 % of Sect. 2.4.
        No guides when omitted, as in the published figure.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is opened when omitted.
    cmap : str, default RADIUS_CMAP
        Sequential colormap for the radius series.
    figsize : tuple of float, default (6.0, 4.0)
        Size of a new figure [in]. Ignored when `ax` is given.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and the axes drawn on.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame.
    ValueError
        If no bias column is found, if `radii` names an absent radius, if
        `clip` does not increase, if `gridsize` is below two, or if no radius
        holds the spread a kernel density needs.

    See Also
    --------
    fluxfootprints.sensor_location_bias_series : Produces `data`.
    fluxfootprints.BIAS_THRESHOLD : The +/-10 % the paper judges against.
    plot_footprint_target_scatter : The same mismatch pair by pair.

    Notes
    -----
    A radius whose biases are all but identical -- a single period, or a
    perfectly homogeneous site -- offers no bandwidth to estimate and is
    skipped rather than drawn as a spike. Its absence from the legend is the
    signal that it was.

    Examples
    --------
    >>> bias = sensor_location_bias_series(pairs, model.x, model.y)  # doctest: +SKIP
    >>> fig, ax = plot_bias_density(bias, threshold=BIAS_THRESHOLD)  # doctest: +SKIP
    >>> ax.set_title("Daytime footprints")                           # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350,
    Sect. 3.3, Fig. 7.
    """
    if clip[1] <= clip[0]:
        raise ValueError(f"clip must increase, got {clip}.")
    if gridsize < 2:
        raise ValueError(f"gridsize must be at least 2, got {gridsize}.")

    frame = _as_frame(data, [radius_column], "data")
    column = _detect_column(frame, _BIAS_COLUMNS, bias_column, "data", "bias")

    radius_values = _numeric(frame[radius_column])
    keep = _selected_radii(radius_values, radii, "data")
    colors = radius_colors(radius_values[np.isfinite(radius_values)], cmap=cmap)
    scale = 100.0 if percent else 1.0

    fig, ax = _axes(ax, figsize)
    grid = np.linspace(clip[0], clip[1], gridsize)
    drawn = 0

    for radius in keep:
        values = _numeric(frame.loc[radius_values == radius, column])
        values = values[np.isfinite(values)] * scale
        if values.size < 2 or np.allclose(values, values[0]):
            # No spread to estimate a bandwidth from; a spike would be a lie.
            continue
        try:
            density = gaussian_kde(values, bw_method=bandwidth)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate covariance
            continue
        ax.plot(
            grid,
            density(grid),
            color=colors[radius],
            linewidth=2.0,
            solid_capstyle="round",
            label=_radius_label(radius),
            zorder=2,
        )
        drawn += 1

    if drawn == 0:
        raise ValueError(
            f"No radius in data holds two or more distinct {column!r} values, "
            "so no kernel density can be estimated."
        )

    ax.axvline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    if threshold is not None:
        for edge in (-threshold * scale, threshold * scale):
            ax.axvline(
                edge, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--"
            )

    ax.set_xlim(*clip)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Sensor location bias (%)" if percent else "Sensor location bias (-)")
    ax.set_ylabel("Kernel density")
    _style(ax)
    ax.legend(loc="best", frameon=False, fontsize="small", handletextpad=0.6)
    return fig, ax


# ------------------------------
# Fig. 5, Fig. 8 -- the three-level index
# ------------------------------


def plot_level_bars(
    data: pd.DataFrame,
    *,
    level_column: str | None = None,
    radius_column: str = "radius",
    radii: Sequence[float] | None = None,
    percent: bool = True,
    ax: Axes | None = None,
    cmap: str = LEVEL_CMAP,
    figsize: tuple[float, float] = (6.0, 4.0),
    width: float = 0.72,
) -> tuple[Figure, Axes]:
    """
    Plot the three-level index stacked across target areas, Fig. 5 and Fig. 8.

    One stacked bar per radius, HIGH at the bottom in the darkest step through
    LOW at the top in the lightest. Over a table of many sites this is the
    paper's Fig. 5 (land cover) and Fig. 8 (vegetation index): the share of
    sites whose footprint still represents the disc, falling as the disc widens.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per site and radius -- the frames of
        :func:`~fluxfootprints.representativeness_summary` concatenated across
        sites, or the site-scope rows of
        :func:`~fluxfootprints.assess_representativeness`, whose ``radius``
        index level is reset here. Rows are counted as given, so subset to one
        ecosystem type, ecoregion, or period first to draw a panel of Fig. 5.
    level_column : str, optional
        Column holding the :class:`~fluxfootprints.Level`. Detected among
        ``"level"``, ``"landcover_level"``, and ``"continuous_level"`` when
        omitted -- ambiguous on a summary table carrying both halves, so name
        the one to draw.
    radius_column : str, default 'radius'
        Column keying the bars.
    radii : sequence of float, optional
        Draw only these radii. Every radius present is drawn when omitted.
    percent : bool, default True
        Scale each bar to 100 %, as the paper's axis does. Set False to stack
        raw site counts, which also shows how many sites each bar rests on.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is opened when omitted.
    cmap : str, default LEVEL_CMAP
        Sequential colormap for the three levels, darkest for HIGH.
    figsize : tuple of float, default (6.0, 4.0)
        Size of a new figure [in]. Ignored when `ax` is given.
    width : float, default 0.72
        Bar width as a fraction of the spacing between radii.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and the axes drawn on. Radii are laid out at even spacing
        and labelled by value, not to scale, as the paper's panels are.

    Raises
    ------
    TypeError
        If `data` is not a DataFrame.
    ValueError
        If no level column is found, if `radii` names an absent radius, if the
        level column holds a value that is not one of ``"high"``, ``"medium"``,
        and ``"low"``, or if every row is missing a level.

    See Also
    --------
    fluxfootprints.representativeness_summary : Produces one site's rows.
    fluxfootprints.Level : The index being counted.
    plot_landcover_composition : The composition behind one site's verdict.

    Notes
    -----
    Rows missing a level are dropped before counting, which is the paper's own
    treatment: 34 of its 214 sites carried a land-cover product that disagreed
    with the site metadata and were left out of the land-cover panels, and a
    site with fewer than :data:`~fluxfootprints.MIN_MATCHES` matched scenes has
    no regression to classify. With ``percent=True`` each bar is therefore a
    share of the sites classified at *that* radius, which need not be the same
    count from bar to bar.

    Examples
    --------
    >>> table = pd.concat(summaries)                          # doctest: +SKIP
    >>> fig, ax = plot_level_bars(table, level_column="landcover_level")
    ... # doctest: +SKIP
    >>> ax.set_title("Land-cover representativeness")         # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350,
    Fig. 5, Fig. 8.
    """
    frame = _as_frame(data, [radius_column], "data")
    column = _detect_column(frame, _LEVEL_COLUMNS, level_column, "data", "level")

    rows = frame[frame[column].notna()]
    if rows.empty:
        raise ValueError(
            f"Every row of data is missing {column!r}, so there is no verdict "
            "to count."
        )

    known = {level.value for level in _LEVEL_ORDER}
    labels = rows[column].astype(str).to_numpy()
    unknown = sorted(set(labels) - known)
    if unknown:
        raise ValueError(
            f"data[{column!r}] holds the value(s) {unknown}, which are not "
            f"levels; expected {[level.value for level in _LEVEL_ORDER]}."
        )

    radius_values = _numeric(rows[radius_column])
    keep = _selected_radii(radius_values, radii, "data")
    colors = level_colors(cmap=cmap)

    counts = {
        level: np.array(
            [
                float(((radius_values == radius) & (labels == level.value)).sum())
                for radius in keep
            ]
        )
        for level in _LEVEL_ORDER
    }
    totals = sum(counts.values())
    if percent:
        counts = {
            level: np.divide(
                100.0 * value, totals, out=np.zeros_like(value), where=totals > 0
            )
            for level, value in counts.items()
        }

    fig, ax = _axes(ax, figsize)
    positions = np.arange(len(keep), dtype=float)
    bottom = np.zeros(len(keep), dtype=float)

    for level in _LEVEL_ORDER:
        heights = counts[level]
        ax.bar(
            positions,
            heights,
            width=width,
            bottom=bottom,
            color=colors[level],
            # A surface-coloured gap, rather than a dark border, separates the
            # segments of a stack.
            edgecolor="white",
            linewidth=1.5,
            label=level.value.capitalize(),
            zorder=2,
        )
        bottom = bottom + heights

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{radius:g}" for radius in keep])
    ax.set_xlabel("Target area around tower (m)")
    ax.set_ylabel("Percentage of sites (%)" if percent else "Number of sites")
    if percent:
        ax.set_ylim(0.0, 100.0)
    _style(ax, axis="y")

    # Square swatches read the stack better than the bar patches' own handles.
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=8, color=colors[level])
        for level in _LEVEL_ORDER
    ]
    # Stacked bars leave no room inside the axes, so the key sits above them.
    ax.legend(
        handles,
        [level.value.capitalize() for level in _LEVEL_ORDER],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=len(_LEVEL_ORDER),
        frameon=False,
        fontsize="small",
        handletextpad=0.4,
        columnspacing=1.4,
    )
    return fig, ax
