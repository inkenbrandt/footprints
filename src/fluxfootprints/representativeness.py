# src/fluxfootprints/representativeness.py
"""
representativeness.py
=====================
Footprint-to-target-area representativeness analysis after Chu et al. (2021).

This module evaluates how well a flux footprint climatology represents the
land-surface conditions of a *target area* -- a fixed-radius disc around the
tower of the kind used as a model grid cell or a remote-sensing pixel window
in synthesis studies. It follows the method of:

    Chu, H., et al. (2021). Representativeness of Eddy-Covariance flux
    footprints for areas surrounding AmeriFlux sites. *Agricultural and Forest
    Meteorology*, **301-302**, 108350.
    https://doi.org/10.1016/j.agrformet.2021.108350

The analysis has three parts, mirroring the paper:

1. **Climatology metrics** (Sect. 2.2). The climatology is truncated at the
   80 % source-weight contour, and summarised by fetch (X80), area (A80),
   symmetry (S80, Eq. 1), and the seasonal and day-night overlap indices
   (O80, Eqs. 2-3).
2. **Categorical evaluation** (Sect. 2.4). The footprint-weighted land-cover
   composition is compared against the target-area composition with a
   chi-square test, and reduced to a three-level representativeness index.
3. **Continuous evaluation** (Sect. 2.4). A footprint-weighted vegetation
   index (EVI in the paper, Eq. 5) is compared against the target-area mean;
   per-period sensor location bias (Eq. 6, after Schmid and Lloyd, 1999) and
   a site-level model II regression (Eq. 7) give a second three-level index.

Grid conventions follow the rest of the package: ``x`` and ``y`` are cell-centre
offsets in metres from the tower, which sits at the origin, and footprint
weights are densities [m⁻²] that must be multiplied by the cell area to obtain
a source fraction.

Notes
-----
The climatologies produced by :class:`~fluxfootprints.FFPModel` and
:class:`~fluxfootprints.ffp_climatology_new` are deliberately *not* rescaled to
integrate to unity over the domain, so their total mass is the mean captured
fraction (< 1). Fractions passed to the contour routines here are therefore
fractions of the full footprint, consistent with the paper, rather than
fractions of the captured mass.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .base_footprint_model import BaseFootprintModel, _source_weight_threshold
from .openet_masking import GridGeometry

__all__ = [
    "TARGET_RADII",
    "ASYMMETRY_THRESHOLD",
    "Level",
    "ClimatologyMetrics",
    "CategoricalResult",
    "ContinuousResult",
    "contour_level_for_fraction",
    "footprint_contour_mask",
    "truncate_to_contour",
    "target_area_mask",
    "footprint_fetch",
    "footprint_area",
    "symmetry_index",
    "footprint_symmetry",
    "climatology_metrics",
    "overlap",
    "seasonal_overlap",
    "daynight_overlap",
    "seasonal_overlap_index",
    "daynight_overlap_index",
    "footprint_weighted_mean",
    "footprint_weighted_composition",
    "target_area_mean",
    "target_area_composition",
    "sensor_location_bias",
    "model2_regression",
    "classify_categorical",
    "classify_continuous",
    "evaluate_landcover",
    "evaluate_vegetation_index",
    "evaluate_representativeness",
    "representativeness_summary",
    "sample_raster_on_grid",
    "predict_sigmav",
    "export_representativeness_gpkg",
]


# ------------------------------
# Constants
# ------------------------------

#: Target-area radii evaluated in Chu et al. (2021), Sect. 2.1 [m].
TARGET_RADII: tuple[int, ...] = (250, 500, 1000, 1500, 2000, 3000)

#: Source-weight fraction at which footprint climatologies are truncated
#: (Chu et al., 2021, Sect. 2.2).
DEFAULT_CONTOUR_FRACTION: float = 0.8

#: Symmetry index below which Chu et al. (2021) call a footprint
#: climatology relatively asymmetric (Sect. 2.2) [-].
ASYMMETRY_THRESHOLD: float = 0.30

#: Significance level for the land-cover composition chi-square test.
DEFAULT_ALPHA: float = 0.05

#: Sensor location bias threshold |Δ| considered representative, Sect. 2.4 [-].
DEFAULT_BIAS_THRESHOLD: float = 0.10

#: Default name of the dimension the monthly climatologies of a site-year are
#: stacked over, for the overlap indices of Eqs. 2-3.
_MONTH_DIM: str = "month"

#: How far a month's weights may stray from unit sum before the overlap
#: indices refuse them [-].
_NORMALIZATION_TOLERANCE: float = 1e-6


class Level(str, Enum):
    """
    Three-level footprint-to-target-area representativeness index.

    The members are ``str`` subclasses, so ``Level.HIGH == "high"`` is True and
    the values serialise directly into DataFrames, CSVs, and GeoPackage fields.

    Attributes
    ----------
    HIGH : Level
        Footprint and target area agree closely. For land cover: a single class
        holds >= 80 % of both the footprint-weighted and the target-area
        composition, and the chi-square test is not significant. For a
        continuous field: R² >= 0.8 with slope 1.0 +/- 0.1 and intercept
        0.0 +/- 0.1.
    MEDIUM : Level
        Partial agreement. For land cover: a single class holds >= 50 % of both
        compositions and the chi-square test is not significant, but the HIGH
        criteria are not met. For a continuous field: R² >= 0.6 and p < 0.05,
        but the HIGH criteria are not met.
    LOW : Level
        Poor agreement. For land cover: no class reaches 50 % in the footprint
        or the target area, or the compositions differ significantly. For a
        continuous field: R² < 0.6 or p >= 0.05.

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ------------------------------
# Optional dependencies
# ------------------------------

#: Install hints surfaced by :func:`_require` when an optional import fails.
_OPTIONAL_DEPENDENCIES: dict[str, str] = {
    "geopandas": "pip install geopandas",
    "rioxarray": "pip install rioxarray",
    "sklearn": "pip install scikit-learn  (or: pip install 'fluxfootprints[contours]')",
}


def _require(module: str) -> ModuleType:
    """
    Import an optional dependency, or raise with install instructions.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"rioxarray"``. Keys of
        :data:`_OPTIONAL_DEPENDENCIES` carry a tailored install hint; any other
        name falls back to ``pip install <module>``.

    Returns
    -------
    types.ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module is not installed, with the install command in the message.

    Examples
    --------
    >>> rio = _require("rioxarray")  # doctest: +SKIP
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        hint = _OPTIONAL_DEPENDENCIES.get(module, f"pip install {module}")
        raise ImportError(
            f"The optional dependency '{module}' is required for this function "
            f"but is not installed.\n    {hint}"
        ) from exc


# ------------------------------
# Result containers
# ------------------------------


@dataclass(frozen=True)
class ClimatologyMetrics:
    """
    Summary metrics of a truncated footprint climatology (Sect. 2.2).

    Attributes
    ----------
    fraction : float
        Source-weight fraction at which the climatology was truncated, e.g.
        0.8 for the 80 % contour.
    contour_level : float
        Footprint density [m⁻²] enclosing `fraction` of the total source
        weight; usable directly as a ``levels`` argument to ``ax.contour``.
    fetch : float
        X80, the maximum distance from the tower to the truncation contour [m].
    area : float
        A80, the area enclosed by the truncation contour [m²].
    symmetry : float
        S80 = A80 / (pi * X80²), Eq. 1. Ranges 0-1; 1 is a perfectly circular
        climatology centred on the tower, and below
        :data:`ASYMMETRY_THRESHOLD` (0.30) the paper calls the climatology
        relatively asymmetric.
    enclosed_fraction : float
        Source weight actually enclosed by the contour, relative to the total
        mass on the domain. Falls below `fraction` when footprint mass leaves
        the domain.
    n_cells : int
        Number of grid cells inside the contour.
    seasonal_overlap : float or None
        O80_season, Eq. 2, the geometric-mean overlap of monthly climatologies
        across a site-year. ``None`` when a single climatology was supplied.
    daynight_overlap : float or None
        O80_daynight, Eq. 3, the mean overlap of paired daytime and nighttime
        climatologies. ``None`` when no pairing was supplied.

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.2.
    """

    fraction: float
    contour_level: float
    fetch: float
    area: float
    symmetry: float
    enclosed_fraction: float
    n_cells: int
    seasonal_overlap: float | None = None
    daynight_overlap: float | None = None


@dataclass(frozen=True)
class CategoricalResult:
    """
    Land-cover representativeness for one target-area radius (Sect. 2.4).

    Attributes
    ----------
    radius : float
        Target-area radius from the tower [m].
    dominant_class : Any
        Land-cover code holding the largest footprint-weighted share. Keys are
        whatever the classification raster carries, typically ``int`` NLCD codes.
    p_footprint : float
        Percentage of `dominant_class` within the footprint, footprint-weighted
        (P_footprint in the paper) [%].
    p_target : float
        Percentage of the same class within the target area (P_target) [%].
    chi2 : float
        Chi-square statistic comparing the two compositions.
    p_value : float
        p-value of the chi-square test. Values >= 0.05 indicate the
        compositions are not significantly different.
    dof : int
        Degrees of freedom of the chi-square test.
    level : Level
        Three-level representativeness index from :func:`classify_categorical`.
    footprint_composition : Mapping
        Full footprint-weighted composition, class code -> percentage [%].
    target_composition : Mapping
        Full target-area composition, class code -> percentage [%].

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """

    radius: float
    dominant_class: Any
    p_footprint: float
    p_target: float
    chi2: float
    p_value: float
    dof: int
    level: Level
    footprint_composition: Mapping[Any, float]
    target_composition: Mapping[Any, float]


@dataclass(frozen=True)
class ContinuousResult:
    """
    Continuous-field representativeness for one target-area radius (Sect. 2.4).

    Holds the site-level model II regression of target-area against
    footprint-weighted values (Eq. 7), reported as in Table 1 of the paper,
    together with the per-period sensor location biases (Eq. 6).

    Attributes
    ----------
    radius : float
        Target-area radius from the tower [m].
    intercept : float
        Regression intercept, beta_0.
    slope : float
        Regression slope, beta_1. A slope below 1 means the footprint saw
        systematically higher values than the target area.
    r_squared : float
        Coefficient of determination.
    p_value : float
        Significance of the regression.
    rmse : float
        Root mean square error between footprint-weighted and target-area values.
    mae : float
        Mean absolute error between footprint-weighted and target-area values.
    n : int
        Number of matched footprint / field periods entering the regression.
    level : Level
        Three-level representativeness index from :func:`classify_continuous`.
    bias : numpy.ndarray
        Per-period sensor location bias Delta (Eq. 6), as a fraction rather
        than a percentage. Length `n`.
    within_threshold : float
        Fraction of periods with ``|Delta| <=`` the bias threshold [-].

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    Schmid, H. P., and Lloyd, C. R. (1999). Agric. For. Meteorol., 93, 195-209.
    """

    radius: float
    intercept: float
    slope: float
    r_squared: float
    p_value: float
    rmse: float
    mae: float
    n: int
    level: Level
    bias: np.ndarray
    within_threshold: float


# ------------------------------
# Contours and masks
# ------------------------------


def _grid_spacing(
    da: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
) -> tuple[float, float]:
    """
    Resolve the grid spacing of a footprint array.

    Parameters
    ----------
    da : xarray.DataArray
        Array carrying ``x`` and ``y`` cell-centre coordinates [m].
    dx, dy : float, optional
        Spacing to use as given; only the missing one is inferred.

    Returns
    -------
    tuple of float
        ``(dx, dy)`` in metres, taken as the median coordinate step so that a
        stray duplicate or trimmed edge cell cannot skew the result.

    Raises
    ------
    ValueError
        If a spacing must be inferred but the corresponding coordinate is
        missing, shorter than two cells, or not positively spaced; or if a
        supplied spacing is not positive and finite.
    """
    spacings: list[float] = []
    for axis, given in (("x", dx), ("y", dy)):
        if given is not None:
            if not np.isfinite(given) or given <= 0:
                raise ValueError(f"d{axis} must be positive and finite, got {given!r}.")
            spacings.append(float(given))
            continue

        if axis not in da.coords:
            raise ValueError(
                f"Cannot infer d{axis}: the array carries no '{axis}' "
                f"coordinate. Pass d{axis} explicitly."
            )
        values = np.asarray(da.coords[axis].values, dtype=float)
        if values.size < 2:
            raise ValueError(
                f"Cannot infer d{axis} from a single '{axis}' cell. "
                f"Pass d{axis} explicitly."
            )
        step = float(np.median(np.abs(np.diff(values))))
        if not np.isfinite(step) or step <= 0:
            raise ValueError(
                f"Cannot infer d{axis}: the '{axis}' coordinate is not "
                f"regularly spaced. Pass d{axis} explicitly."
            )
        spacings.append(step)

    return spacings[0], spacings[1]


def _inside_cells(w: xr.DataArray) -> xr.DataArray:
    """
    Reduce a contour mask or a truncated climatology to an inside-mask.

    Parameters
    ----------
    w : xarray.DataArray
        Either a boolean mask from :func:`footprint_contour_mask`, or a
        climatology truncated by :func:`truncate_to_contour`, renormalised
        or not.

    Returns
    -------
    xarray.DataArray
        Boolean array with the dims and coords of `w`.

    Notes
    -----
    A truncated climatology zeroes every cell outside the contour and keeps
    the retained ones at or above a strictly positive contour level, so
    "finite and positive" recovers exactly the cells the truncation kept.
    """
    values = np.asarray(w.values)
    if values.dtype == np.bool_:
        inside = values
    else:
        floats = values.astype(float, copy=False)
        inside = np.isfinite(floats) & (floats > 0.0)
    return xr.DataArray(inside, coords=w.coords, dims=w.dims)


def contour_level_for_fraction(
    fclim: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
) -> float:
    """
    Find the footprint density enclosing a given fraction of the source weight.

    Sorts the climatology in descending order and accumulates ``value * dx * dy``
    until `fraction` is reached, following the same construction as
    :meth:`~fluxfootprints.FFPModel.get_source_area_contour`.

    Parameters
    ----------
    fclim : xarray.DataArray
        Footprint climatology with dims ``(x, y)`` and density units [m⁻²].
    dx, dy : float, optional
        Grid spacing [m]. Inferred from the ``x``/``y`` coordinates when omitted.
    fraction : float, default 0.8
        Source-weight fraction to enclose, in (0, 1).

    Returns
    -------
    float
        Footprint density [m⁻²] at the contour. Pass as ``levels`` to
        ``matplotlib.axes.Axes.contour``.

    Raises
    ------
    ValueError
        If `fraction` is outside (0, 1), or `fclim` carries no positive mass.

    Notes
    -----
    Because the package's climatologies integrate to the mean captured fraction
    rather than to 1, a `fraction` larger than the captured mass saturates at
    the smallest positive density on the grid rather than raising.
    """
    cell_dx, cell_dy = _grid_spacing(fclim, dx, dy)
    return _source_weight_threshold(fclim.values, cell_dx * cell_dy, fraction)


def footprint_contour_mask(
    fclim: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
) -> xr.DataArray:
    """
    Build a boolean mask of the cells inside a source-weight contour.

    Parameters
    ----------
    fclim : xarray.DataArray
        Footprint climatology with dims ``(x, y)`` [m⁻²].
    dx, dy : float, optional
        Grid spacing [m]. Inferred from coordinates when omitted.
    fraction : float, default 0.8
        Source-weight fraction to enclose.

    Returns
    -------
    xarray.DataArray
        Boolean array with the dims and coords of `fclim`, True inside the
        contour.

    See Also
    --------
    contour_level_for_fraction : The threshold this mask is built from.
    truncate_to_contour : Mask and renormalise in one step.

    Notes
    -----
    Non-finite cells never enter the mask, so a climatology padded with NaN
    outside its domain is handled like one padded with zeros.
    """
    level = contour_level_for_fraction(fclim, dx, dy, fraction)
    mask = (fclim >= level) & np.isfinite(fclim)
    mask.attrs = {
        "long_name": f"{fraction:g} source-weight contour mask",
        "contour_fraction": float(fraction),
        "contour_level": level,
    }
    mask.name = "contour_mask"
    return mask


def truncate_to_contour(
    fclim: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    renormalize: bool = True,
) -> xr.DataArray:
    """
    Truncate a climatology at a source-weight contour and rescale its weights.

    Chu et al. (2021), Sect. 2.2, truncate every climatology at the 80 %
    contour and rescale the retained weights to sum to unity, so that all
    downstream weighted statistics are taken over the same source area.

    Parameters
    ----------
    fclim : xarray.DataArray
        Footprint climatology with dims ``(x, y)`` [m⁻²].
    dx, dy : float, optional
        Grid spacing [m]. Inferred from coordinates when omitted.
    fraction : float, default 0.8
        Source-weight fraction to retain.
    renormalize : bool, default True
        If True, rescale the retained cells so they sum to 1, giving unitless
        weights suitable for :func:`footprint_weighted_mean`. If False, retain
        the original densities [m⁻²].

    Returns
    -------
    xarray.DataArray
        Climatology with cells outside the contour set to zero, carrying the
        dims and coords of `fclim`. Unitless when `renormalize` is True.

    Notes
    -----
    The rescaling is a sum over cells, not an area integral, matching the
    paper's ``sum(phi_ik) = 1`` convention. The retained cells therefore sum to
    1 regardless of `fraction`: the fraction selects the source area, it does
    not survive as the mass of the result.

    The mask is applied with ``where(..., 0.0)``, so cells outside the contour
    -- including any non-finite ones -- come back as zeros rather than NaN, and
    the weights stay safe to sum.
    """
    mask = footprint_contour_mask(fclim, dx, dy, fraction)
    level = float(mask.attrs["contour_level"])
    n_cells = int(mask.sum())

    truncated = fclim.where(mask, 0.0)

    if renormalize:
        total = float(truncated.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                "The cells inside the contour carry no positive weight, so the "
                "climatology cannot be renormalised."
            )
        truncated = truncated / total

    truncated.attrs = dict(fclim.attrs)
    truncated.attrs.update(
        {
            "contour_fraction": float(fraction),
            "contour_level": level,
            "contour_n_cells": n_cells,
            "contour_renormalized": str(bool(renormalize)).lower(),
        }
    )
    if renormalize:
        truncated.attrs["units"] = "1"
    truncated.name = fclim.name
    return truncated


def target_area_mask(
    x: np.ndarray | xr.DataArray,
    y: np.ndarray | xr.DataArray,
    radius: float,
) -> xr.DataArray:
    """
    Build a boolean mask of the disc of a given radius around the tower.

    Parameters
    ----------
    x, y : numpy.ndarray or xarray.DataArray
        Cell-centre offsets from the tower [m], as produced by the footprint
        models (``model.x``, ``model.y``).
    radius : float
        Target-area radius [m], e.g. one of :data:`TARGET_RADII`.

    Returns
    -------
    xarray.DataArray
        Boolean array with dims ``(x, y)``, True where the cell centre lies
        within `radius` of the origin.

    Raises
    ------
    ValueError
        If `radius` is not positive.

    Notes
    -----
    Membership is decided on cell centres, so a target area larger than the
    model domain is silently clipped to the domain. Compare the mask's cell
    count against ``pi * radius**2 / (dx * dy)`` to detect that case.
    """
    raise NotImplementedError


# ------------------------------
# Climatology metrics (Sect. 2.2)
# ------------------------------


def footprint_fetch(
    mask: xr.DataArray,
    origin: tuple[float, float] = (0.0, 0.0),
) -> float:
    """
    Compute the footprint fetch X80, Sect. 2.2.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean contour mask with dims ``(x, y)``, as returned by
        :func:`footprint_contour_mask`. A climatology truncated by
        :func:`truncate_to_contour` is equally accepted, renormalised or not:
        its positive cells are exactly the cells inside the contour.
    origin : tuple of float, default (0.0, 0.0)
        Tower position ``(x, y)`` in the coordinates of `mask` [m]. The
        package's grids are tower-centred, so the default holds unless the
        array carries coordinates that put the tower somewhere else.

    Returns
    -------
    float
        Maximum distance from the tower to a cell inside the contour [m].
        ``nan`` if no cell is inside.

    Raises
    ------
    ValueError
        If `mask` carries no ``x`` or ``y`` coordinate to measure distance from.

    Notes
    -----
    Distances are measured to cell *centres*, as in :func:`target_area_mask`,
    so the fetch understates the true reach of the contour by up to half a cell
    diagonal.
    """
    for axis in ("x", "y"):
        if axis not in mask.coords:
            raise ValueError(
                f"Cannot measure fetch: the mask carries no '{axis}' "
                f"coordinate to measure distance from."
            )

    inside = _inside_cells(mask)
    if not bool(inside.any()):
        return float("nan")

    distance = np.hypot(mask.coords["x"] - origin[0], mask.coords["y"] - origin[1])
    return float(distance.where(inside).max())


def footprint_area(
    mask: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
) -> float:
    """
    Compute the footprint area A80, Sect. 2.2.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean contour mask with dims ``(x, y)``. A climatology truncated by
        :func:`truncate_to_contour` is equally accepted; see
        :func:`footprint_fetch`.
    dx, dy : float, optional
        Grid spacing [m]. Inferred from the mask coordinates when omitted.

    Returns
    -------
    float
        Area enclosed by the contour [m²], as the cell count times ``dx * dy``.
        ``0.0`` if no cell is inside.

    Notes
    -----
    Counting whole cells makes the area a step function of the grid spacing:
    a contour is resolved to within one cell of its true extent, so coarse
    grids and small source areas do not mix.
    """
    cell_dx, cell_dy = _grid_spacing(mask, dx, dy)
    return float(_inside_cells(mask).sum()) * cell_dx * cell_dy


def symmetry_index(area: float, fetch: float) -> float:
    """
    Compute the footprint symmetry index S80, Eq. 1.

    .. math:: S_{80} = \\frac{A_{80}}{\\pi X_{80}^2}

    Parameters
    ----------
    area : float
        Footprint area A80 [m²].
    fetch : float
        Footprint fetch X80 [m].

    Returns
    -------
    float
        Symmetry index in [0, 1]. One indicates a perfectly circular
        climatology centred on the tower; the paper flags values below
        :data:`ASYMMETRY_THRESHOLD` (0.30) as relatively asymmetric, arising
        from uni- or bimodal prevailing winds. ``nan`` if `fetch` is zero or
        non-finite, or if `area` is non-finite.

    See Also
    --------
    footprint_symmetry : The same index straight from a truncated climatology.

    Notes
    -----
    The disc of radius X80 the index compares against is the smallest one
    centred on the tower that contains the contour, so the ratio is bounded
    above by 1 in the continuum. On a grid it can creep past 1, because the
    area counts whole cells while the fetch reaches only to cell centres; the
    result is clipped to [0, 1] rather than reported above one.
    """
    if not np.isfinite(area) or not np.isfinite(fetch) or fetch <= 0.0:
        return float("nan")
    return float(min(max(area / (np.pi * fetch**2), 0.0), 1.0))


def footprint_symmetry(
    w: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
) -> float:
    """
    Compute the footprint symmetry index S80 of a truncated climatology, Eq. 1.

    Convenience wrapper that takes the fetch and area off one array and feeds
    them to :func:`symmetry_index`.

    Parameters
    ----------
    w : xarray.DataArray
        Truncated climatology from :func:`truncate_to_contour`, or the boolean
        mask from :func:`footprint_contour_mask`.
    dx, dy : float, optional
        Grid spacing [m]. Inferred from coordinates when omitted.
    origin : tuple of float, default (0.0, 0.0)
        Tower position ``(x, y)`` in the coordinates of `w` [m].

    Returns
    -------
    float
        Symmetry index in [0, 1]; values below :data:`ASYMMETRY_THRESHOLD`
        (0.30) are the paper's relatively asymmetric climatologies. ``nan`` if
        `w` holds no cell inside the contour.
    """
    return symmetry_index(
        footprint_area(w, dx, dy),
        footprint_fetch(w, origin=origin),
    )


def climatology_metrics(
    fclim: xr.DataArray,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    seasonal_overlap: float | None = None,
    daynight_overlap: float | None = None,
) -> ClimatologyMetrics:
    """
    Summarise a footprint climatology with the metrics of Sect. 2.2.

    Parameters
    ----------
    fclim : xarray.DataArray
        Footprint climatology with dims ``(x, y)`` [m⁻²]. An array already
        truncated by :func:`truncate_to_contour` is also accepted and is
        summarised on the contour it was cut at, rather than being truncated a
        second time; see Notes.
    dx, dy : float, optional
        Grid spacing [m]. Inferred from coordinates when omitted.
    fraction : float, default 0.8
        Source-weight fraction defining the truncation contour. Ignored for an
        already-truncated `fclim`, whose own fraction is used instead.
    seasonal_overlap, daynight_overlap : float, optional
        Precomputed overlap indices to carry into the result, from
        :func:`seasonal_overlap` and :func:`daynight_overlap`.
        Both are site-year properties spanning several climatologies and so
        cannot be derived from `fclim` alone.

    Returns
    -------
    ClimatologyMetrics
        Fetch, area, symmetry, cell count, contour level, and enclosed mass.
        Symmetry below :data:`ASYMMETRY_THRESHOLD` (0.30) is what the paper
        calls a relatively asymmetric climatology.

    Notes
    -----
    Truncated input is recognised by the ``contour_fraction`` attribute that
    :func:`truncate_to_contour` writes, and its fraction and contour level are
    read back from there. ``enclosed_fraction`` is then ``nan``: the mass
    outside the contour has been zeroed, so what share of the domain's source
    weight was retained is no longer recoverable from the array. Pass the raw
    climatology to get it.

    Examples
    --------
    >>> from fluxfootprints import build_climatology, climatology_metrics
    >>> model = build_climatology(df)                     # doctest: +SKIP
    >>> climatology_metrics(model.fclim_2d).symmetry      # doctest: +SKIP
    0.52
    """
    cell_dx, cell_dy = _grid_spacing(fclim, dx, dy)

    if "contour_fraction" in fclim.attrs:
        mask = _inside_cells(fclim)
        level = float(fclim.attrs["contour_level"])
        fraction = float(fclim.attrs["contour_fraction"])
        enclosed = float("nan")
    else:
        mask = footprint_contour_mask(fclim, cell_dx, cell_dy, fraction)
        level = float(mask.attrs["contour_level"])
        finite = fclim.where(np.isfinite(fclim), 0.0)
        total = float(finite.sum())
        enclosed = (
            float(finite.where(mask, 0.0).sum() / total)
            if total > 0.0
            else float("nan")
        )

    fetch = footprint_fetch(mask)
    area = footprint_area(mask, cell_dx, cell_dy)

    return ClimatologyMetrics(
        fraction=float(fraction),
        contour_level=level,
        fetch=fetch,
        area=area,
        symmetry=symmetry_index(area, fetch),
        enclosed_fraction=enclosed,
        n_cells=int(mask.sum()),
        seasonal_overlap=(
            None if seasonal_overlap is None else float(seasonal_overlap)
        ),
        daynight_overlap=(
            None if daynight_overlap is None else float(daynight_overlap)
        ),
    )


def _clamp_unit(value: float) -> float:
    """
    Clip an overlap index into [0, 1].

    Parameters
    ----------
    value : float
        Index value, bounded by 0 and 1 in exact arithmetic for weights that
        sum to 1.

    Returns
    -------
    float
        `value` clipped into [0, 1].

    Notes
    -----
    Only floating-point noise -- of order 1e-16 on a sum of square roots --
    can push a valid index past the bound, because the callers check that
    every month sums to 1 before summing anything. The clip keeps the
    documented range exact rather than papering over an unnormalised input.
    """
    return float(min(max(value, 0.0), 1.0))


def _weight_values(w: xr.DataArray | np.ndarray, name: str) -> np.ndarray:
    """
    Validate one footprint's weights and return them as a float array.

    Parameters
    ----------
    w : xarray.DataArray or numpy.ndarray
        Footprint weights.
    name : str
        How to refer to `w` in the error messages.

    Returns
    -------
    numpy.ndarray
        The weights as floats, with the shape of `w`.

    Raises
    ------
    ValueError
        If `w` holds no cells, or if any weight is non-finite or negative.
    """
    values = np.asarray(w, dtype=float)
    if values.size == 0:
        raise ValueError(f"{name} holds no cells.")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} holds non-finite weights. Truncation zeroes the cells "
            f"outside the contour rather than masking them; see "
            f"truncate_to_contour."
        )
    if np.any(values < 0.0):
        raise ValueError(
            f"{name} holds negative weights, which are not source weights."
        )
    return values


def _check_same_grid(
    a: xr.DataArray | np.ndarray,
    b: xr.DataArray | np.ndarray,
    name_a: str,
    name_b: str,
) -> None:
    """
    Require two footprints to sit on the same grid, cell for cell.

    Parameters
    ----------
    a, b : xarray.DataArray or numpy.ndarray
        Footprints to compare. Dims and coordinates are compared when both are
        DataArrays; shapes always are.
    name_a, name_b : str
        How to refer to `a` and `b` in the error messages.

    Raises
    ------
    ValueError
        If the dims, the shapes, or the coordinates along any dim differ.

    Notes
    -----
    The overlap kernel multiplies the two arrays, and xarray would quietly
    align mismatched coordinates first -- an inner join that drops cells and
    an outer join that invents NaN ones. Comparing up front turns that into an
    error instead.
    """
    if isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray):
        if a.dims != b.dims:
            raise ValueError(
                f"{name_a} and {name_b} are on different grids: dims "
                f"{a.dims} and {b.dims}."
            )
        for axis in a.dims:
            in_a, in_b = axis in a.coords, axis in b.coords
            if not in_a and not in_b:
                continue
            if in_a != in_b:
                raise ValueError(
                    f"{name_a} and {name_b} disagree over the '{axis}' "
                    f"coordinate: only one of them carries it."
                )
            if not np.array_equal(
                np.asarray(a.coords[axis].values),
                np.asarray(b.coords[axis].values),
            ):
                raise ValueError(
                    f"{name_a} and {name_b} are on different grids: their "
                    f"'{axis}' coordinates differ."
                )
    if np.shape(a) != np.shape(b):
        raise ValueError(
            f"{name_a} and {name_b} are on different grids: shapes "
            f"{np.shape(a)} and {np.shape(b)}."
        )


def _check_normalized(w: xr.DataArray | np.ndarray, name: str) -> None:
    """
    Require a footprint's weights to sum to one.

    Parameters
    ----------
    w : xarray.DataArray or numpy.ndarray
        Footprint weights, expected to be renormalised.
    name : str
        How to refer to `w` in the error message.

    Raises
    ------
    ValueError
        If the weights do not sum to 1 within
        :data:`_NORMALIZATION_TOLERANCE`.

    Notes
    -----
    This is what bounds the overlap indices by 1: for weights summing to 1,
    Cauchy-Schwarz gives ``sum sqrt(w1 w2) <= sqrt(sum w1 sum w2) = 1``. A
    non-finite sum compares False and slips through here, to be caught by
    :func:`_weight_values` with a message about the actual problem.
    """
    total = float(np.asarray(w, dtype=float).sum())
    if abs(total - 1.0) > _NORMALIZATION_TOLERANCE:
        raise ValueError(
            f"{name} sums to {total:.6g}, not 1. The overlap indices are "
            f"defined on climatologies rescaled to unit sum; see "
            f"truncate_to_contour."
        )


def _months_dim(w: xr.DataArray, dim: str, name: str) -> int:
    """
    Require a stacked array to carry the month dimension, and size it.

    Parameters
    ----------
    w : xarray.DataArray
        Monthly climatologies stacked over `dim`.
    dim : str
        Name of the stacking dimension.
    name : str
        How to refer to `w` in the error messages.

    Returns
    -------
    int
        Number of months, K.

    Raises
    ------
    TypeError
        If `w` is not a DataArray.
    ValueError
        If `w` carries no `dim` dimension, or holds no months along it.
    """
    if not isinstance(w, xr.DataArray):
        raise TypeError(
            f"{name} must be an xarray.DataArray with the months stacked over "
            f"'{dim}', got {type(w).__name__}."
        )
    if dim not in w.dims:
        raise ValueError(
            f"{name} carries no '{dim}' dimension to stack the months over; "
            f"its dims are {w.dims}."
        )
    if w.sizes[dim] == 0:
        raise ValueError(f"{name} holds no months along '{dim}'.")
    return int(w.sizes[dim])


def _stack_months(
    climatologies: Sequence[xr.DataArray],
    name: str,
    dim: str = _MONTH_DIM,
) -> xr.DataArray:
    """
    Stack a sequence of monthly climatologies over a new month dimension.

    Parameters
    ----------
    climatologies : sequence of xarray.DataArray
        Monthly climatologies on a common grid, one per month.
    name : str
        How to refer to the sequence in the error messages.
    dim : str, default "month"
        Name of the dimension to stack over.

    Returns
    -------
    xarray.DataArray
        The climatologies stacked over `dim`, in the order given.

    Raises
    ------
    ValueError
        If the sequence is empty, if a member already carries `dim` as a
        dimension, or if the grids differ.
    """
    if isinstance(climatologies, xr.DataArray):
        raise TypeError(
            f"{name} must be a sequence of DataArrays, one per month; pass a "
            f"stacked array to seasonal_overlap or daynight_overlap instead."
        )
    months = list(climatologies)
    if not months:
        raise ValueError(f"{name} holds no climatologies.")

    first = months[0]
    for index, month in enumerate(months):
        if isinstance(month, xr.DataArray) and dim in month.dims:
            raise ValueError(
                f"{name}[{index}] already carries a '{dim}' dimension; pass "
                f"the stacked array to seasonal_overlap or daynight_overlap "
                f"instead of a sequence."
            )
        if index:
            _check_same_grid(first, month, f"{name}[0]", f"{name}[{index}]")

    return xr.concat(months, dim=dim, join="exact")


def overlap(
    w1: xr.DataArray | np.ndarray,
    w2: xr.DataArray | np.ndarray,
) -> float:
    """
    Compute the overlap of two footprints, the kernel of Eqs. 2-3.

    .. math:: \\sum_{i=1}^{I} \\sqrt{\\varphi^{1}_{i} \\varphi^{2}_{i}}

    The cell-wise geometric mean of two footprints, summed over cells: the
    Bhattacharyya coefficient of the two weight fields read as discrete
    distributions. It is Eq. 2 for the special case of two months, and the
    per-month term of Eq. 3.

    Parameters
    ----------
    w1, w2 : xarray.DataArray or numpy.ndarray
        Footprint weights on a common grid, each summing to 1 -- the output of
        :func:`truncate_to_contour`. DataArrays are compared cell for cell
        rather than aligned on their coordinates, so a mismatched grid raises
        instead of silently joining.

    Returns
    -------
    float
        Overlap in [0, 1] for weights that sum to 1: one when the two
        footprints are identical, zero when their supports are disjoint.

    Raises
    ------
    ValueError
        If the two grids differ in dims, shape, or coordinates; or if either
        array is empty or holds a non-finite or negative weight.

    See Also
    --------
    seasonal_overlap : Eq. 2, over the K months of a site-year.
    daynight_overlap : Eq. 3, over paired daytime and nighttime months.

    Notes
    -----
    Normalisation is left to the callers rather than checked here, so that the
    kernel stays usable on any pair of non-negative fields; the upper bound of
    1 holds only for weights that sum to 1. :func:`seasonal_overlap` and
    :func:`daynight_overlap` do check it.
    """
    _check_same_grid(w1, w2, "w1", "w2")
    values1 = _weight_values(w1, "w1")
    values2 = _weight_values(w2, "w2")
    return float(np.sqrt(values1 * values2).sum())


def seasonal_overlap(weights: xr.DataArray, dim: str = _MONTH_DIM) -> float:
    """
    Compute the seasonal footprint overlap index O80_season, Eq. 2.

    .. math:: O_{80,season} = \\sum_{i=1}^{I}
              \\left( \\prod_{k=1}^{K} \\varphi_{ik} \\right)^{1/K}

    The cell-wise geometric mean of the `K` monthly climatologies in a
    site-year, summed over the `I` grid cells.

    Parameters
    ----------
    weights : xarray.DataArray
        Monthly climatologies stacked over `dim`, on a common grid and each
        month already rescaled to sum to 1 -- the output of
        :func:`truncate_to_contour`. Computed separately for daytime and
        nighttime in the paper.
    dim : str, default "month"
        Name of the dimension the months are stacked over.

    Returns
    -------
    float
        Overlap index in [0, 1]. One indicates perfectly overlapping monthly
        climatologies; the paper treats values below 0.8 as showing
        "noticeable monthly variability", which it found at 32-44 % of the
        studied site-years, concentrated in the cropland, grassland, and
        wetland sites whose canopy height swings through the growing season.

    Raises
    ------
    TypeError
        If `weights` is not a DataArray.
    ValueError
        If `weights` carries no `dim` dimension or holds fewer than two months
        along it; if any weight is non-finite or negative; or if any month
        does not sum to 1.

    See Also
    --------
    overlap : The two-footprint kernel, which Eq. 2 reduces to at K = 2.
    seasonal_overlap_index : The same index from a sequence of climatologies.
    daynight_overlap : The companion index across daytime and nighttime.

    Notes
    -----
    The exponent used here is 1/K, a true geometric mean over all K months.
    Equation 2 as printed in Chu et al. (2021) carries 1/k inside a product
    running over k = 1..K, which is a typo: the index k has no value outside
    the product it is bound to, and only the 1/K reading returns 1.0 when
    every month is identical -- the defining property of an overlap index.

    The geometric mean is computed as ``exp(mean(log(w)))`` over the cells
    every month covers, and is taken as zero elsewhere, so a zero in any one
    month propagates to a zero cell rather than a NaN. The index therefore
    measures the source area common to *all* months, and a single month
    pointing elsewhere is enough to drive it to zero.
    """
    n_months = _months_dim(weights, dim, "weights")
    if n_months < 2:
        raise ValueError(
            f"The seasonal overlap index compares months, so it needs at "
            f"least two, got {n_months}."
        )

    values = _weight_values(weights.transpose(dim, ...), "weights")
    for index in range(n_months):
        _check_normalized(values[index], f"weights month {index}")

    common = np.all(values > 0.0, axis=0)
    if not common.any():
        return 0.0
    logs = np.log(np.where(common, values, 1.0))
    geometric_mean = np.where(common, np.exp(logs.mean(axis=0)), 0.0)
    return _clamp_unit(float(geometric_mean.sum()))


def daynight_overlap(
    day: xr.DataArray,
    night: xr.DataArray,
    dim: str = _MONTH_DIM,
) -> float:
    """
    Compute the daytime-nighttime footprint overlap index O80_daynight, Eq. 3.

    .. math:: O_{80,daynight} = \\frac{1}{K} \\sum_{k=1}^{K} \\sum_{i=1}^{I}
              \\left( \\varphi^{day}_{ik} \\varphi^{night}_{ik} \\right)^{1/2}

    The :func:`overlap` of the paired daytime and nighttime climatologies of
    each month, averaged over the `K` months of a site-year.

    Parameters
    ----------
    day, night : xarray.DataArray
        Paired monthly daytime and nighttime climatologies stacked over `dim`,
        on a common grid and each month already rescaled to sum to 1 -- the
        output of :func:`truncate_to_contour`.
    dim : str, default "month"
        Name of the dimension the months are stacked over. The two arrays are
        paired by position along it; where both carry a coordinate for it, the
        coordinates must agree.

    Returns
    -------
    float
        Overlap index in [0, 1]. One indicates perfectly overlapping day and
        night climatologies; around 93 % of the site-years in the paper
        exceeded 0.8, because a climatology aggregates many half-hours from
        many wind directions and the day and night source areas share their
        high-weight core even where the nighttime one reaches farther.

    Raises
    ------
    TypeError
        If `day` or `night` is not a DataArray.
    ValueError
        If either array carries no `dim` dimension or holds no months along
        it; if the two hold different numbers of months or disagree over the
        `dim` coordinate; if their grids differ; if any weight is non-finite
        or negative; or if any month does not sum to 1.

    See Also
    --------
    overlap : The per-month kernel this averages.
    daynight_overlap_index : The same index from sequences of climatologies.
    seasonal_overlap : The companion index across the months of a site-year.

    Notes
    -----
    Unlike Eq. 2, this index pairs the two climatologies month by month before
    averaging, so a month whose day and night footprints diverge lowers the
    result without zeroing it. Nighttime climatologies reached about 45 %
    farther and covered about 90 % more area than daytime ones at more than
    95 % of the site-years in the paper.
    """
    n_months = _months_dim(day, dim, "day")
    n_nights = _months_dim(night, dim, "night")
    if n_months != n_nights:
        raise ValueError(
            f"day and night hold different numbers of months along '{dim}': "
            f"{n_months} and {n_nights}."
        )
    if (
        dim in day.coords
        and dim in night.coords
        and not np.array_equal(
            np.asarray(day.coords[dim].values), np.asarray(night.coords[dim].values)
        )
    ):
        raise ValueError(
            f"day and night carry different '{dim}' coordinates, so the "
            f"months cannot be paired by position."
        )

    total = 0.0
    for index in range(n_months):
        month_day = day.isel({dim: index})
        month_night = night.isel({dim: index})
        _check_normalized(month_day, f"day month {index}")
        _check_normalized(month_night, f"night month {index}")
        total += overlap(month_day, month_night)

    return _clamp_unit(total / n_months)


def seasonal_overlap_index(climatologies: Sequence[xr.DataArray]) -> float:
    """
    Compute the seasonal footprint overlap index O80_season, Eq. 2.

    Adapter over :func:`seasonal_overlap` for monthly climatologies held as a
    sequence rather than stacked over a dimension. See that function for the
    equation, the 1/K exponent, and the handling of zeros.

    Parameters
    ----------
    climatologies : sequence of xarray.DataArray
        Monthly climatologies for one site-year, each with dims ``(x, y)`` on a
        common grid and each summing to 1. Computed separately for daytime and
        nighttime in the paper.

    Returns
    -------
    float
        Overlap index in [0, 1]. One indicates perfectly overlapping monthly
        climatologies; the paper treats values below 0.8 as showing
        "noticeable monthly variability".

    Raises
    ------
    ValueError
        If fewer than two climatologies are supplied, or their grids differ.

    Notes
    -----
    The geometric mean is zero wherever any single month has zero weight, so
    the index is dominated by the area common to *all* months.
    """
    return seasonal_overlap(_stack_months(climatologies, "climatologies"))


def daynight_overlap_index(
    daytime: Sequence[xr.DataArray],
    nighttime: Sequence[xr.DataArray],
) -> float:
    """
    Compute the daytime-nighttime footprint overlap index O80_daynight, Eq. 3.

    Adapter over :func:`daynight_overlap` for monthly climatologies held as
    sequences rather than stacked over a dimension. See that function for the
    equation.

    Parameters
    ----------
    daytime, nighttime : sequence of xarray.DataArray
        Paired monthly daytime and nighttime climatologies for one site-year,
        each with dims ``(x, y)`` on a common grid and each summing to 1.

    Returns
    -------
    float
        Overlap index in [0, 1]. One indicates perfectly overlapping day and
        night climatologies; around 93 % of the site-years in the paper
        exceeded 0.8.

    Raises
    ------
    ValueError
        If the two sequences differ in length, or their grids differ.
    """
    return daynight_overlap(
        _stack_months(daytime, "daytime"),
        _stack_months(nighttime, "nighttime"),
    )


# ------------------------------
# Weighted and target-area statistics
# ------------------------------


def footprint_weighted_mean(
    weights: xr.DataArray,
    field: xr.DataArray,
) -> float:
    """
    Compute a footprint-weighted mean of a continuous field, Eq. 5.

    .. math:: EVI_{footprint} = \\sum_{j=1}^{J} \\varphi_j EVI_j

    Parameters
    ----------
    weights : xarray.DataArray
        Truncated, renormalised footprint weights with dims ``(x, y)``, summing
        to 1. See :func:`truncate_to_contour`.
    field : xarray.DataArray
        Continuous land-surface field on the same grid, e.g. Landsat EVI.

    Returns
    -------
    float
        Footprint-weighted mean of `field`. ``nan`` if no cell has both a
        positive weight and a finite field value.

    Notes
    -----
    Cells where `field` is NaN are dropped and the remaining weights are
    renormalised, so partial raster coverage biases the result toward the
    covered portion of the source area rather than returning NaN outright.
    """
    raise NotImplementedError


def footprint_weighted_composition(
    weights: xr.DataArray,
    classes: xr.DataArray,
) -> dict[Any, float]:
    """
    Compute the footprint-weighted composition of a categorical field.

    Parameters
    ----------
    weights : xarray.DataArray
        Truncated, renormalised footprint weights with dims ``(x, y)``.
    classes : xarray.DataArray
        Categorical land-cover raster on the same grid, typically integer codes
        such as the consolidated NLCD / Land Cover of Canada groups of Table S6.

    Returns
    -------
    dict
        Class code -> footprint-weighted percentage [%], summing to 100 over
        the classes present. NaN cells in `classes` are excluded and the
        remaining weights renormalised.

    See Also
    --------
    target_area_composition : The unweighted counterpart over a target disc.
    """
    raise NotImplementedError


def target_area_mean(
    field: xr.DataArray,
    radius: float,
) -> float:
    """
    Compute the unweighted mean of a continuous field over a target area.

    Parameters
    ----------
    field : xarray.DataArray
        Continuous land-surface field with dims ``(x, y)`` on the tower-centred
        grid.
    radius : float
        Target-area radius [m].

    Returns
    -------
    float
        Arithmetic mean of `field` over cells within `radius` of the tower
        (EVI_target in the paper), ignoring NaN. ``nan`` if no finite cell falls
        inside the disc.
    """
    raise NotImplementedError


def target_area_composition(
    classes: xr.DataArray,
    radius: float,
) -> dict[Any, float]:
    """
    Compute the composition of a categorical field over a target area.

    Parameters
    ----------
    classes : xarray.DataArray
        Categorical land-cover raster with dims ``(x, y)`` on the tower-centred
        grid.
    radius : float
        Target-area radius [m].

    Returns
    -------
    dict
        Class code -> percentage [%] of the target-area cells, summing to 100.
        NaN cells are excluded from both numerator and denominator.
    """
    raise NotImplementedError


# ------------------------------
# Bias and regression (Sect. 2.4)
# ------------------------------


def sensor_location_bias(
    footprint_value: float | np.ndarray,
    target_value: float | np.ndarray,
) -> float | np.ndarray:
    """
    Compute the sensor location bias Delta, Eq. 6.

    .. math:: \\Delta = \\frac{EVI_{footprint} - EVI_{target}}{EVI_{target}}

    After Schmid and Lloyd (1999); the time-explicit footprint-to-target-area
    bias for one period.

    Parameters
    ----------
    footprint_value : float or numpy.ndarray
        Footprint-weighted value(s) from :func:`footprint_weighted_mean`.
    target_value : float or numpy.ndarray
        Target-area mean value(s) from :func:`target_area_mean`.

    Returns
    -------
    float or numpy.ndarray
        Relative bias as a fraction, not a percentage; multiply by 100 to match
        the paper's figures. Positive values mean the footprint covered higher
        values than its surroundings, which held at every target radius in the
        paper. ``nan`` where `target_value` is zero or non-finite.

    References
    ----------
    Schmid, H. P., and Lloyd, C. R. (1999). Spatial representativeness and the
    location bias of flux footprints over inhomogeneous areas. *Agric. For.
    Meteorol.*, **93**, 195-209.
    """
    raise NotImplementedError


def model2_regression(
    footprint_values: np.ndarray | Sequence[float],
    target_values: np.ndarray | Sequence[float],
) -> tuple[float, float, float, float]:
    """
    Fit the site-level model II regression of Eq. 7.

    .. math:: EVI_{target} \\sim \\beta_0 + \\beta_1 EVI_{footprint}

    The paper uses R's ``lmodel2``, appropriate here because both variables
    carry error. This implements the reduced major axis (standard major axis)
    solution, whose slope is the ordinary least-squares slope divided by the
    Pearson correlation coefficient.

    Parameters
    ----------
    footprint_values : array_like
        Footprint-weighted values, the predictor.
    target_values : array_like
        Target-area means, the response. Must match `footprint_values` in length.

    Returns
    -------
    intercept : float
        beta_0.
    slope : float
        beta_1.
    r_squared : float
        Squared Pearson correlation of the two inputs.
    p_value : float
        Two-sided significance of the correlation.

    Raises
    ------
    ValueError
        If the inputs differ in length, or fewer than three finite pairs remain.

    Notes
    -----
    Pairs where either value is non-finite are dropped before fitting. The RMA
    slope is undefined in sign when the correlation is zero; the sign is taken
    from the correlation, as ``lmodel2`` does.
    """
    raise NotImplementedError


# ------------------------------
# Classification (Sect. 2.4)
# ------------------------------


def classify_categorical(
    p_footprint: float,
    p_target: float,
    p_value: float,
    alpha: float = DEFAULT_ALPHA,
) -> Level:
    """
    Assign the land-cover representativeness level, Sect. 2.4.

    The 50 % and 80 % criteria follow Goeckede et al. (2008).

    Parameters
    ----------
    p_footprint : float
        Percentage of the dominant land-cover type within the footprint [%].
    p_target : float
        Percentage of the same type within the target area [%].
    p_value : float
        Chi-square p-value comparing the two compositions.
    alpha : float, default 0.05
        Significance level; compositions with ``p_value >= alpha`` are treated
        as not significantly different.

    Returns
    -------
    Level
        ``HIGH`` when both percentages are >= 80 and the compositions do not
        differ significantly; ``MEDIUM`` when both are >= 50 and the
        compositions do not differ significantly; ``LOW`` otherwise.

    Examples
    --------
    >>> classify_categorical(92.0, 88.0, 0.42)   # doctest: +SKIP
    <Level.HIGH: 'high'>
    >>> classify_categorical(92.0, 61.0, 0.31)   # doctest: +SKIP
    <Level.MEDIUM: 'medium'>
    """
    raise NotImplementedError


def classify_continuous(
    r_squared: float,
    slope: float,
    intercept: float,
    p_value: float,
    alpha: float = DEFAULT_ALPHA,
) -> Level:
    """
    Assign the continuous-field representativeness level, Sect. 2.4.

    Parameters
    ----------
    r_squared : float
        Coefficient of determination of the model II regression.
    slope : float
        Regression slope beta_1.
    intercept : float
        Regression intercept beta_0.
    p_value : float
        Significance of the regression.
    alpha : float, default 0.05
        Significance level for the MEDIUM criterion.

    Returns
    -------
    Level
        ``HIGH`` when ``r_squared >= 0.8`` with ``0.9 <= slope <= 1.1`` and
        ``-0.1 <= intercept <= 0.1``; ``MEDIUM`` when ``r_squared >= 0.6`` and
        ``p_value < alpha``; ``LOW`` otherwise.

    Notes
    -----
    The intercept tolerance is absolute and calibrated to EVI's 0-1 range. A
    field on a different scale, e.g. land surface temperature in kelvin, needs
    a rescaled criterion before this classification is meaningful.
    """
    raise NotImplementedError


# ------------------------------
# Site-level evaluation
# ------------------------------


def evaluate_landcover(
    fclim: xr.DataArray,
    classes: xr.DataArray,
    radii: Sequence[float] = TARGET_RADII,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    alpha: float = DEFAULT_ALPHA,
) -> list[CategoricalResult]:
    """
    Evaluate land-cover representativeness across a series of target areas.

    Identifies the dominant land-cover type within the footprint, compares its
    share and the full composition against each target area with a chi-square
    test, and classifies the result (Sect. 2.4).

    Parameters
    ----------
    fclim : xarray.DataArray
        Footprint climatology with dims ``(x, y)`` [m⁻²].
    classes : xarray.DataArray
        Categorical land-cover raster on the same grid. Use
        :func:`sample_raster_on_grid` to bring an external product onto it.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m].
    dx, dy : float, optional
        Grid spacing [m]. Inferred from coordinates when omitted.
    fraction : float, default 0.8
        Source-weight fraction defining the truncation contour.
    alpha : float, default 0.05
        Significance level for the chi-square test.

    Returns
    -------
    list of CategoricalResult
        One entry per radius, in the order given.

    Raises
    ------
    ValueError
        If `fclim` and `classes` are not on the same grid.

    Notes
    -----
    The chi-square test compares footprint-weighted counts against target-area
    counts; classes absent from both are dropped. The paper found 34 of 214
    sites where the land-cover product disagreed with the site's own IGBP
    metadata, so a disagreement between this result and site metadata is worth
    checking against the product before trusting it.
    """
    raise NotImplementedError


def evaluate_vegetation_index(
    climatologies: Sequence[xr.DataArray],
    fields: Sequence[xr.DataArray],
    radii: Sequence[float] = TARGET_RADII,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    bias_threshold: float = DEFAULT_BIAS_THRESHOLD,
) -> list[ContinuousResult]:
    """
    Evaluate continuous-field representativeness across a series of target areas.

    For each matched climatology / field pair the footprint-weighted value
    (Eq. 5), target-area mean, and sensor location bias (Eq. 6) are computed;
    the pairs are then regressed per radius (Eq. 7) and classified.

    Parameters
    ----------
    climatologies : sequence of xarray.DataArray
        Footprint climatologies with dims ``(x, y)`` [m⁻²], one per period. In
        the paper these are monthly climatologies matched to the retrieval date
        of the underlying Landsat scene.
    fields : sequence of xarray.DataArray
        Continuous fields on the same grid, matched element-wise to
        `climatologies`, e.g. Landsat EVI.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m].
    dx, dy : float, optional
        Grid spacing [m]. Inferred from coordinates when omitted.
    fraction : float, default 0.8
        Source-weight fraction defining the truncation contour.
    alpha : float, default 0.05
        Significance level for the regression.
    bias_threshold : float, default 0.10
        ``|Delta|`` below which a period counts as representative, reported as
        ``ContinuousResult.within_threshold``.

    Returns
    -------
    list of ContinuousResult
        One entry per radius, in the order given.

    Raises
    ------
    ValueError
        If the two sequences differ in length, or fewer than three pairs are
        supplied.

    Notes
    -----
    The paper required at least six matched pairs for a site-level regression,
    with a median of 13 per site. Fewer pairs will fit but the resulting level
    should not be trusted.
    """
    raise NotImplementedError


def evaluate_representativeness(
    model: BaseFootprintModel,
    landcover: xr.DataArray | None = None,
    vegetation_index: Sequence[xr.DataArray] | None = None,
    climatologies: Sequence[xr.DataArray] | None = None,
    radii: Sequence[float] = TARGET_RADII,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    bias_threshold: float = DEFAULT_BIAS_THRESHOLD,
) -> tuple[list[CategoricalResult], list[ContinuousResult]]:
    """
    Run the full representativeness analysis for a fitted footprint model.

    Convenience driver over :func:`evaluate_landcover` and
    :func:`evaluate_vegetation_index`, taking grid geometry and the climatology
    from `model` in the same style as
    :func:`~fluxfootprints.summarize_periods`.

    Parameters
    ----------
    model : BaseFootprintModel
        A model on which ``run()`` has already been called, supplying
        ``fclim_2d``, ``x``, ``y``, ``dx``, and ``dy``.
    landcover : xarray.DataArray, optional
        Categorical land-cover raster on the model grid. Skipped when omitted.
    vegetation_index : sequence of xarray.DataArray, optional
        Continuous fields, one per period. Skipped when omitted.
    climatologies : sequence of xarray.DataArray, optional
        Per-period climatologies matched to `vegetation_index`. Defaults to
        repeating ``model.fclim_2d`` for every field, which is correct only
        when the fields all share one source period.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m].
    fraction : float, default 0.8
        Source-weight fraction defining the truncation contour.
    alpha : float, default 0.05
        Significance level for both classifications.
    bias_threshold : float, default 0.10
        Sensor location bias threshold.

    Returns
    -------
    categorical : list of CategoricalResult
        Land-cover results, empty when `landcover` is omitted.
    continuous : list of ContinuousResult
        Continuous-field results, empty when `vegetation_index` is omitted.

    Raises
    ------
    RuntimeError
        If `model` has not been run, i.e. ``fclim_2d`` is None.

    Examples
    --------
    >>> from fluxfootprints import build_climatology, evaluate_representativeness
    >>> model = build_climatology(df, model_type="ffp")          # doctest: +SKIP
    >>> cat, cont = evaluate_representativeness(model, landcover=nlcd)  # doctest: +SKIP
    """
    raise NotImplementedError


def representativeness_summary(
    categorical: Sequence[CategoricalResult] | None = None,
    continuous: Sequence[ContinuousResult] | None = None,
    metrics: ClimatologyMetrics | None = None,
    site_id: str | None = None,
) -> pd.DataFrame:
    """
    Flatten representativeness results into a tidy table.

    Parameters
    ----------
    categorical : sequence of CategoricalResult, optional
        Land-cover results from :func:`evaluate_landcover`.
    continuous : sequence of ContinuousResult, optional
        Continuous-field results from :func:`evaluate_vegetation_index`.
    metrics : ClimatologyMetrics, optional
        Climatology metrics, repeated across every row when supplied.
    site_id : str, optional
        Site identifier written to a ``site_id`` column, e.g. ``"US-MOz"``.

    Returns
    -------
    pandas.DataFrame
        One row per target radius, with columns for the land-cover statistics
        and level, the regression statistics and level, and the climatology
        metrics. Missing halves are filled with NaN, so the frame can be
        concatenated across sites and written with
        :func:`~fluxfootprints.export_contour_stats_csv` conventions.
    """
    raise NotImplementedError


# ------------------------------
# Optional-dependency helpers
# ------------------------------


def sample_raster_on_grid(
    raster_path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    station_lat: float,
    station_lon: float,
    crs: str | int | None = "auto",
    band: int = 1,
    categorical: bool = False,
) -> xr.DataArray:
    """
    Read an external raster onto the tower-centred footprint grid.

    Georeferences the grid with
    :func:`~fluxfootprints.footprint_grid_geometry`, then reprojects the raster
    onto it, so that land-cover and vegetation-index products can be compared
    cell-for-cell against a climatology.

    Parameters
    ----------
    raster_path : str or pathlib.Path
        Path to a raster readable by rasterio / rioxarray, e.g. an NLCD tile or
        a Landsat EVI scene.
    x, y : numpy.ndarray
        Cell-centre offsets from the tower [m] (``model.x``, ``model.y``).
    station_lat, station_lon : float
        Tower position in decimal degrees (WGS 84).
    crs : str, int, or None, default "auto"
        Target CRS for the grid; ``"auto"`` selects the local WGS 84 UTM zone,
        matching :func:`~fluxfootprints.export_rasters_geotiff`.
    band : int, default 1
        One-based band index to read.
    categorical : bool, default False
        If True, resample with nearest neighbour to preserve class codes; if
        False, resample bilinearly for a continuous field.

    Returns
    -------
    xarray.DataArray
        Raster values with dims ``(x, y)`` and the coords of the footprint
        grid, NaN where the source raster does not cover the grid.

    Raises
    ------
    ImportError
        If ``rioxarray`` is not installed.
    ValueError
        If the raster has no CRS, so it cannot be reprojected.

    See Also
    --------
    fluxfootprints.footprint_grid_geometry : Builds the target :class:`GridGeometry`.
    fluxfootprints.openet_mask_on_grid : The same reprojection for data-availability masks.
    """
    raise NotImplementedError


def predict_sigmav(
    df: pd.DataFrame,
    estimator: Any | None = None,
    predictors: Sequence[str] | None = None,
) -> pd.Series:
    """
    Gap-fill the crosswind velocity standard deviation, Sect. 2.2 and Text S1.

    V_SIGMA is missing at many towers but is required by the FFP model. The
    paper trains a random forest on the 106 sites that do report it, using
    friction velocity, boundary-layer height, wind speed, incoming shortwave
    radiation, the Obukhov stability parameter, and IGBP class, and reports
    R² = 0.79 with MAE = 0.15 m s⁻¹ against a withheld test set.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing the predictor columns and, for rows used in
        training, an observed ``sigmav`` column [m s⁻¹].
    estimator : object, optional
        A fitted scikit-learn regressor to apply. When omitted, a
        ``RandomForestRegressor`` is fitted on the rows of `df` that have an
        observed ``sigmav``.
    predictors : sequence of str, optional
        Predictor column names. Defaults to the paper's six:
        ``["ustar", "h", "umean", "sw_in", "zm_over_ol", "igbp"]``.

    Returns
    -------
    pandas.Series
        Sigma_v [m s⁻¹] indexed like `df`, with observed values retained where
        present and predictions filled in elsewhere.

    Raises
    ------
    ImportError
        If ``scikit-learn`` is not installed and no `estimator` is supplied.
    ValueError
        If required predictor columns are missing, or no rows carry an
        observed ``sigmav`` to train on.

    Notes
    -----
    Predicted sigma_v adds uncertainty mainly in the crosswind dimension of the
    footprint. The paper argues this largely cancels when aggregating many
    timesteps into a monthly climatology, but advises against relying on it for
    half-hourly footprint analysis.
    """
    raise NotImplementedError


def export_representativeness_gpkg(
    results: pd.DataFrame,
    geometry: GridGeometry,
    gpkg_path: str | Path,
    layer: str = "representativeness",
) -> Path:
    """
    Write representativeness results to a GeoPackage as target-area discs.

    Parameters
    ----------
    results : pandas.DataFrame
        Tidy results from :func:`representativeness_summary`, carrying a
        ``radius`` column.
    geometry : GridGeometry
        Georeferencing for the footprint grid, from
        :func:`~fluxfootprints.footprint_grid_geometry`.
    gpkg_path : str or pathlib.Path
        Output GeoPackage path.
    layer : str, default "representativeness"
        Layer name to write.

    Returns
    -------
    pathlib.Path
        The written path.

    Raises
    ------
    ImportError
        If ``geopandas`` is not installed.
    ValueError
        If `results` carries no ``radius`` column.

    Notes
    -----
    Each row becomes a buffered point centred on the tower with the row's
    radius, so the layer draws as nested discs attributed with their
    representativeness levels.
    """
    raise NotImplementedError
