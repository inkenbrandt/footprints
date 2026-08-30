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

import datetime as dt
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chisquare, pearsonr
from scipy.stats import t as _student_t

from .base_footprint_model import BaseFootprintModel, _source_weight_threshold
from .openet_masking import GridGeometry, footprint_grid_geometry

__all__ = [
    "TARGET_RADII",
    "ASYMMETRY_THRESHOLD",
    "BIAS_THRESHOLD",
    "MIN_MATCHES",
    "Level",
    "ClimatologyMetrics",
    "WeightedValue",
    "CategoricalResult",
    "ContinuousResult",
    "RMAFit",
    "contour_level_for_fraction",
    "footprint_contour_mask",
    "truncate_to_contour",
    "target_area_mask",
    "potential_radiation",
    "partition_daynight",
    "monthly_climatologies",
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
    "footprint_weighted_value",
    "footprint_weighted_composition",
    "target_area_value",
    "target_area_composition",
    "sensor_location_bias",
    "sensor_location_bias_series",
    "rma_regression",
    "model2_regression",
    "classify_categorical",
    "classify_continuous",
    "categorical_representativeness",
    "continuous_representativeness",
    "evaluate_landcover",
    "evaluate_vegetation_index",
    "representativeness_summary",
    "sample_raster_on_grid",
    "predict_sigmav",
    "export_representativeness_gpkg",
    "RESULT_INDEX",
    "RESULT_COLUMNS",
    "CHU_TABLE_COLUMNS",
    "assess_representativeness",
    "representativeness_table",
    "export_representativeness_tables",
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

#: Sensor location bias threshold ``|Δ|`` considered representative,
#: Sect. 2.4 [-].
#: Chu et al. (2021) adopt the 10 % of Chen et al. (2011) and Kim et al. (2006).
BIAS_THRESHOLD: float = 0.10

#: Matched periods Chu et al. (2021) required before fitting a site-level
#: regression, Sect. 2.4; 166 of the 214 sites cleared it.
MIN_MATCHES: int = 6

#: Default name of the dimension the monthly climatologies of a site-year are
#: stacked over, for the overlap indices of Eqs. 2-3.
_MONTH_DIM: str = "month"

#: How far a month's weights may stray from unit sum before the overlap
#: indices refuse them [-].
_NORMALIZATION_TOLERANCE: float = 1e-6

#: Total solar irradiance at one astronomical unit, the scale of the potential
#: incoming shortwave radiation of Sect. 2.2 [W m-2].
SOLAR_CONSTANT: float = 1361.0

#: Column searched for a precomputed potential incoming shortwave radiation,
#: matched case-insensitively. The AmeriFlux BASE name.
SW_IN_POT_COLUMN: str = "SW_IN_POT"

#: Name of the dimension separating the daytime and nighttime climatologies of
#: a month.
_PERIOD_DIM: str = "period"

#: Coordinate labels along :data:`_PERIOD_DIM`.
DAYTIME: str = "daytime"
NIGHTTIME: str = "nighttime"
ALL_HOURS: str = "all"

#: Accepted values of the ``partition`` argument of
#: :func:`monthly_climatologies`.
_PARTITIONS: tuple[str, ...] = ("month", "month+daynight")

#: Names a results Dataset may hold its time-resolved footprints under, in
#: search order.
_FOOTPRINT_VARIABLES: tuple[str, ...] = ("footprint_2d", "f_2d")


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
    "ee": "pip install earthengine-api  (or: pip install 'fluxfootprints[gee]')",
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
        ``nan`` for an already-truncated climatology that no longer carries
        the level it was cut at, such as one sliced out of the stack that
        :func:`monthly_climatologies` returns.
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
class WeightedValue:
    """
    A field value averaged over a footprint or a target area (Sect. 2.4).

    Returned by :func:`footprint_weighted_value` and :func:`target_area_value`,
    which differ only in the weights they average under: the footprint's own
    source weights, or the uniform weights of a target-area disc.

    Attributes
    ----------
    value : float
        The averaged field value -- EVI_footprint of Eq. 5, or EVI_target.
        ``nan`` when no cell carried both weight and data.
    retained_weight : float
        Fraction of the weight that fell on cells holding data, in [0, 1],
        before the renormalisation that produced `value`. It is 1 for a raster
        covering the whole source area, and below 1 where the raster is nodata
        or does not reach -- the cue that `value` describes only part of the
        intended area. Over a target area, where every cell of the disc weighs
        the same, it is the fraction of the disc's cells that held data.
    n_cells : int
        Number of cells that contributed to `value`.

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """

    value: float
    retained_weight: float
    n_cells: int


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
    level : Level or None
        Three-level representativeness index from :func:`classify_continuous`,
        or None where fewer than three periods survived at this radius and no
        regression could be fitted.
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
    level: Level | None
    bias: np.ndarray
    within_threshold: float


@dataclass(frozen=True)
class RMAFit:
    """
    A reduced major axis fit and its confidence intervals.

    What :func:`rma_regression` returns: the model II regression of Eq. 7,
    with the interval estimates the paper's ``lmodel2`` reports alongside the
    point estimates of Table 1.

    Attributes
    ----------
    slope : float
        beta_1, the RMA slope. ``nan`` when either variable is constant.
    intercept : float
        beta_0, ``mean(y) - slope * mean(x)``.
    r_squared : float
        Squared Pearson correlation of the two inputs. Note that this is a
        property of the correlation alone: it does not describe the scatter
        about the RMA line, and it is the same whichever variable is called
        the predictor.
    p_value : float
        Two-sided significance of that correlation, testing rho = 0. The RMA
        slope has no null of its own -- it cannot be zero -- so this is the
        test the paper's MEDIUM criterion uses.
    n : int
        Number of finite pairs the fit rests on.
    slope_ci : tuple of float
        Lower and upper confidence limits on `slope`, in that order.
    intercept_ci : tuple of float
        Lower and upper confidence limits on `intercept`, in that order.
    ci_level : float
        Confidence level the two intervals were computed at, e.g. 0.95.
    ci_method : str
        ``"analytical"`` or ``"bootstrap"``, the estimator used for them.

    See Also
    --------
    rma_regression : Produces this fit.
    classify_continuous : Turns it into the three-level index of Sect. 2.4.

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Eq. 7.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    n: int
    slope_ci: tuple[float, float]
    intercept_ci: tuple[float, float]
    ci_level: float
    ci_method: str


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
        weights suitable for :func:`footprint_weighted_value`. If False, retain
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
        If `radius` is not positive and finite, or if `x` or `y` is not a
        non-empty 1-D array of cell-centre offsets.

    Notes
    -----
    Membership is decided on cell centres, so a target area larger than the
    model domain is silently clipped to the domain. Compare the mask's cell
    count against ``pi * radius**2 / (dx * dy)`` to detect that case.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be positive and finite, got {radius!r}.")

    offsets: list[np.ndarray] = []
    for axis, values in (("x", x), ("y", y)):
        coords = np.asarray(values, dtype=float)
        if coords.ndim != 1 or coords.size == 0:
            raise ValueError(
                f"{axis} must be a non-empty 1-D array of cell-centre offsets "
                f"from the tower, as the models produce it, got shape "
                f"{coords.shape}. Reduce a meshgrid to its axis first."
            )
        offsets.append(coords)

    x_values, y_values = offsets
    inside = np.hypot(x_values[:, None], y_values[None, :]) <= float(radius)
    return xr.DataArray(
        inside,
        coords={"x": x_values, "y": y_values},
        dims=("x", "y"),
        name="target_area",
        attrs={
            "radius": float(radius),
            "long_name": f"target area within {float(radius):g} m of the tower",
        },
    )


# ------------------------------
# Day-night partitioning and monthly climatologies (Sect. 2.2)
# ------------------------------


def _as_tzinfo(tz: str | float | dt.tzinfo) -> dt.tzinfo:
    """
    Coerce a time-zone specification to a :class:`datetime.tzinfo`.

    Parameters
    ----------
    tz : str, float, or datetime.tzinfo
        A fixed offset from UTC in hours (e.g. ``-7`` for US Mountain Standard
        Time), an IANA zone name (e.g. ``"America/Denver"``), or a ready-made
        ``tzinfo``.

    Returns
    -------
    datetime.tzinfo
        The resolved zone.

    Raises
    ------
    TypeError
        If `tz` is none of the accepted types.
    ValueError
        If a numeric offset is not finite or exceeds 24 h, or if a string is
        not a zone name the system time-zone database knows.

    Notes
    -----
    A numeric offset and a zone name are *not* interchangeable for flux data.
    AmeriFlux BASE timestamps are in local standard time all year, with no
    daylight-saving shift, so the offset is the faithful choice; an IANA name
    would advance the clock by an hour each summer and displace every sunrise
    with it.
    """
    if isinstance(tz, dt.tzinfo):
        return tz
    if isinstance(tz, bool):  # bool is an int; nobody means an offset by it
        raise TypeError(
            f"tz must be a UTC offset in hours, an IANA zone name, or a "
            f"datetime.tzinfo, got {tz!r}."
        )
    if isinstance(tz, (int, float, np.integer, np.floating)):
        hours = float(tz)
        if not np.isfinite(hours) or abs(hours) > 24.0:
            raise ValueError(
                f"A numeric tz is a UTC offset in hours and must be finite and "
                f"within +/- 24, got {tz!r}."
            )
        return dt.timezone(dt.timedelta(hours=hours))
    if isinstance(tz, str):
        try:
            return ZoneInfo(tz)
        except Exception as exc:
            raise ValueError(
                f"tz={tz!r} is not a zone name the time-zone database knows. "
                f"Pass an IANA name such as 'America/Denver', or -- for "
                f"AmeriFlux data, which is in local standard time year-round "
                f"-- a fixed UTC offset in hours such as -7."
            ) from exc
    raise TypeError(
        f"tz must be a UTC offset in hours, an IANA zone name, or a "
        f"datetime.tzinfo, got {type(tz).__name__}."
    )


def _utc_index(
    index: pd.DatetimeIndex,
    tz: str | float | dt.tzinfo | None,
) -> pd.DatetimeIndex:
    """
    Localise a timestamp index and convert it to UTC.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Measurement timestamps, time-zone naive or aware.
    tz : str, float, datetime.tzinfo, or None
        Zone the naive timestamps are in, as accepted by :func:`_as_tzinfo`.
        Must be None for an already-aware index.

    Returns
    -------
    pandas.DatetimeIndex
        The same instants, expressed in UTC.

    Raises
    ------
    TypeError
        If `index` is not a DatetimeIndex.
    ValueError
        If `index` holds a NaT; if it is naive and `tz` is None; or if it is
        aware and `tz` was given as well.

    Notes
    -----
    Refusing a `tz` alongside an aware index is deliberate: silently ignoring
    one of two disagreeing zones is how a whole record ends up an hour off.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"The timestamps must be a pandas.DatetimeIndex, got "
            f"{type(index).__name__}. Set the frame's index to its timestamp "
            f"column first."
        )
    if index.hasnans:
        raise ValueError(
            "The timestamp index holds NaT, so the solar position is "
            "undefined there. Drop those rows first."
        )

    if index.tz is not None:
        if tz is not None:
            raise ValueError(
                f"The timestamp index already carries the time zone "
                f"{index.tz}, so tz={tz!r} would be a second, possibly "
                f"conflicting answer. Pass tz=None."
            )
        return index.tz_convert("UTC")

    if tz is None:
        raise ValueError(
            "The timestamp index is time-zone naive, so the solar position "
            "cannot be placed on the clock. Pass tz -- a fixed UTC offset in "
            "hours for AmeriFlux local standard time (e.g. tz=-7), or an IANA "
            "zone name."
        )
    return index.tz_localize(_as_tzinfo(tz)).tz_convert("UTC")


def _julian_day(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert a UTC timestamp index to Julian days.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps in UTC.

    Returns
    -------
    numpy.ndarray
        Julian day numbers, i.e. days since noon UTC on 1 January 4713 BC in
        the proleptic Julian calendar.

    Notes
    -----
    ``datetime64[ns]`` counts nanoseconds from the Unix epoch, which is Julian
    day 2440587.5, so the conversion is exact arithmetic rather than a
    calendar walk.
    """
    naive = index.tz_localize(None) if index.tz is not None else index
    nanoseconds = np.asarray(naive.values, dtype="datetime64[ns]").astype("int64")
    return nanoseconds / 86_400_000_000_000.0 + 2_440_587.5


def _solar_geometry(
    julian_day: np.ndarray,
    latitude: float,
    longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the solar zenith cosine and the Earth-Sun distance.

    Implements the low-precision solar position algorithm published by NOAA
    (after Meeus, *Astronomical Algorithms*, 2nd ed., Ch. 25), which places
    the Sun to within about 0.01 degrees over 1900-2100 -- far finer than the
    horizon crossing needs.

    Parameters
    ----------
    julian_day : numpy.ndarray
        Julian days of the instants to evaluate, from :func:`_julian_day`.
    latitude : float
        Site latitude, positive north [degrees].
    longitude : float
        Site longitude, positive east [degrees].

    Returns
    -------
    cos_zenith : numpy.ndarray
        Cosine of the solar zenith angle, negative when the Sun is below the
        horizon.
    radius : numpy.ndarray
        Earth-Sun distance [astronomical units], which varies by about +/- 1.7
        % over the year and so moves the top-of-atmosphere flux by about
        +/- 3.4 %.

    Notes
    -----
    The zenith angle returned is geometric: no atmospheric refraction and no
    solar-disc radius are applied. That makes ``cos_zenith > 0`` exactly "the
    Sun's centre is above the astronomical horizon", the convention behind the
    potential-radiation day-night split. Refraction would move each crossing
    earlier or later by a few minutes and, at a half-hourly resolution, would
    reclassify at most one record per sunrise and sunset.
    """
    century = (julian_day - 2451545.0) / 36525.0

    # Geometric mean longitude and mean anomaly of the Sun [rad].
    mean_longitude = np.deg2rad(
        np.mod(280.46646 + century * (36000.76983 + century * 0.0003032), 360.0)
    )
    mean_anomaly = np.deg2rad(357.52911 + century * (35999.05029 - 0.0001537 * century))
    eccentricity = 0.016708634 - century * (0.000042037 + 0.0000001267 * century)

    # Equation of the centre, i.e. true minus mean anomaly [rad].
    centre = np.deg2rad(
        np.sin(mean_anomaly) * (1.914602 - century * (0.004817 + 0.000014 * century))
        + np.sin(2.0 * mean_anomaly) * (0.019993 - 0.000101 * century)
        + np.sin(3.0 * mean_anomaly) * 0.000289
    )
    true_longitude = mean_longitude + centre
    true_anomaly = mean_anomaly + centre
    radius = (
        1.000001018
        * (1.0 - eccentricity**2)
        / (1.0 + eccentricity * np.cos(true_anomaly))
    )

    # Apparent longitude and true obliquity, both corrected for nutation.
    nutation = np.deg2rad(125.04 - 1934.136 * century)
    apparent_longitude = (
        true_longitude - np.deg2rad(0.00569) - np.deg2rad(0.00478) * np.sin(nutation)
    )
    mean_obliquity = np.deg2rad(
        23.0
        + (
            26.0
            + (21.448 - century * (46.815 + century * (0.00059 - century * 0.001813)))
            / 60.0
        )
        / 60.0
    )
    obliquity = mean_obliquity + np.deg2rad(0.00256) * np.cos(nutation)
    declination = np.arcsin(np.sin(obliquity) * np.sin(apparent_longitude))

    # Equation of time [minutes]: apparent minus mean solar time.
    tangent = np.tan(obliquity / 2.0) ** 2
    equation_of_time = 4.0 * np.rad2deg(
        tangent * np.sin(2.0 * mean_longitude)
        - 2.0 * eccentricity * np.sin(mean_anomaly)
        + 4.0
        * eccentricity
        * tangent
        * np.sin(mean_anomaly)
        * np.cos(2.0 * mean_longitude)
        - 0.5 * tangent**2 * np.sin(4.0 * mean_longitude)
        - 1.25 * eccentricity**2 * np.sin(2.0 * mean_anomaly)
    )

    # True solar time [minutes past local solar midnight], then hour angle.
    minutes_utc = np.mod(julian_day + 0.5, 1.0) * 1440.0
    true_solar_time = np.mod(
        minutes_utc + equation_of_time + 4.0 * longitude, 1440.0
    )
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    phi = np.deg2rad(latitude)
    cos_zenith = np.sin(phi) * np.sin(declination) + np.cos(phi) * np.cos(
        declination
    ) * np.cos(hour_angle)
    return cos_zenith, radius


def _check_site(latitude: float, longitude: float) -> tuple[float, float]:
    """
    Validate a site geolocation.

    Parameters
    ----------
    latitude : float
        Site latitude, positive north [degrees].
    longitude : float
        Site longitude, positive east [degrees].

    Returns
    -------
    tuple of float
        ``(latitude, longitude)`` as floats.

    Raises
    ------
    ValueError
        If either is missing or non-finite, if the latitude is outside
        [-90, 90], or if the longitude is outside [-360, 360].
    """
    if latitude is None or longitude is None:
        raise ValueError(
            "The solar position needs the site geolocation: pass latitude and "
            "longitude, or supply a precomputed "
            f"{SW_IN_POT_COLUMN} column."
        )
    lat, lon = float(latitude), float(longitude)
    if not np.isfinite(lat) or abs(lat) > 90.0:
        raise ValueError(
            f"latitude must be finite and within +/- 90, got {latitude!r}."
        )
    if not np.isfinite(lon) or abs(lon) > 360.0:
        raise ValueError(
            f"longitude must be finite and within +/- 360 degrees east, got "
            f"{longitude!r}. Western longitudes are negative."
        )
    return lat, lon


def _timestamps(df: pd.DataFrame | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Take the timestamp index out of a frame, or pass an index through.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.DatetimeIndex
        Measurement records indexed by time, or the timestamps alone.

    Returns
    -------
    pandas.DatetimeIndex
        The timestamps.

    Raises
    ------
    TypeError
        If `df` is neither a DataFrame nor a DatetimeIndex.
    """
    if isinstance(df, pd.DatetimeIndex):
        return df
    if isinstance(df, pd.DataFrame):
        return df.index
    raise TypeError(
        f"Expected a pandas.DataFrame indexed by time or a "
        f"pandas.DatetimeIndex, got {type(df).__name__}."
    )


def _find_column(df: pd.DataFrame | pd.DatetimeIndex, name: str) -> str | None:
    """
    Look up a column by name, ignoring case.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.DatetimeIndex
        Frame to search; an index carries no columns and yields None.
    name : str
        Column name to match case-insensitively.

    Returns
    -------
    str or None
        The matching column's actual name, or None if there is no match.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    target = name.casefold()
    for column in df.columns:
        if isinstance(column, str) and column.casefold() == target:
            return column
    return None


def potential_radiation(
    df: pd.DataFrame | pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    tz: str | float | dt.tzinfo | None = None,
) -> pd.Series:
    """
    Compute potential (top-of-atmosphere) incoming shortwave radiation.

    .. math:: SW_{IN,POT} = \\frac{S_0}{R^2} \\max(\\cos\\theta_z, 0)

    The flux a horizontal surface would receive with no atmosphere: the solar
    constant, scaled by the inverse square of the Earth-Sun distance `R` and
    projected onto the horizontal by the solar zenith angle. This is the
    quantity Chu et al. (2021), Sect. 2.2, threshold at 0 W m⁻² to separate
    daytime from nighttime records.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.DatetimeIndex
        Measurement records indexed by timestamp, or the timestamps alone.
    latitude : float
        Site latitude, positive north [degrees].
    longitude : float
        Site longitude, positive east [degrees]; western longitudes are
        negative.
    tz : str, float, datetime.tzinfo, or None, optional
        Time zone the timestamps are in: a fixed UTC offset in hours
        (e.g. ``-7``), an IANA zone name (e.g. ``"America/Denver"``), or a
        ``tzinfo``. Omit it when the index is already time-zone aware.

    Returns
    -------
    pandas.Series
        Potential radiation [W m⁻²] named ``"SW_IN_POT"``, on the index of
        `df`. Zero whenever the Sun is at or below the horizon.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame or DatetimeIndex, or `tz` is of an
        unusable type.
    ValueError
        If the index holds a NaT; if it is naive and `tz` is None, or aware
        and `tz` was given; or if the geolocation is out of range.

    See Also
    --------
    partition_daynight : The day-night flag this radiation defines.

    Notes
    -----
    Timestamps are read as instants, so a period-averaged record is placed at
    whatever its label says -- the start of the averaging period for
    AmeriFlux ``TIMESTAMP_START``, the end for ``TIMESTAMP_END``. Only the one
    half-hour that straddles each horizon crossing can be classified either
    way by that choice.

    The solar constant used is 1361 W m⁻², the modern total-solar-irradiance
    value; the flux therefore peaks near 1413 W m⁻² at perihelion in early
    January and near 1321 W m⁻² at aphelion in early July.

    Examples
    --------
    Six-hourly instants over the winter solstice at a mountain-west site, in
    local standard time: only local noon has the Sun up.

    >>> import pandas as pd
    >>> times = pd.date_range("2020-12-21", periods=4, freq="6h")
    >>> potential_radiation(times, 40.0, -111.0, tz=-7).round(1).tolist()
    [0.0, 0.0, 624.2, 0.0]
    """
    lat, lon = _check_site(latitude, longitude)
    index = _timestamps(df)
    utc = _utc_index(index, tz)

    cos_zenith, radius = _solar_geometry(_julian_day(utc), lat, lon)
    flux = SOLAR_CONSTANT / radius**2 * np.clip(cos_zenith, 0.0, None)
    return pd.Series(flux, index=index, name=SW_IN_POT_COLUMN)


def partition_daynight(
    df: pd.DataFrame | pd.DatetimeIndex,
    latitude: float | None = None,
    longitude: float | None = None,
    tz: str | float | dt.tzinfo | None = None,
    sw_in_pot: str | None = SW_IN_POT_COLUMN,
) -> pd.Series:
    """
    Split records into daytime and nighttime, after Sect. 2.2.

    Chu et al. (2021) separate daytime from nighttime by the potential
    incoming radiation computed from the site's geolocation and time zone,
    calling a record daytime where that radiation exceeds 0 W m⁻² -- i.e.
    wherever the Sun is above the horizon. Daytime and nighttime footprints
    are then aggregated into separate climatologies, because the nighttime
    ones reach about 45 % farther and cover about 90 % more area.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.DatetimeIndex
        Measurement records indexed by timestamp, or the timestamps alone. A
        frame carrying a potential-radiation column is used as-is; see
        `sw_in_pot`.
    latitude : float, optional
        Site latitude, positive north [degrees]. Required unless the
        radiation is supplied by column.
    longitude : float, optional
        Site longitude, positive east [degrees]; western longitudes are
        negative. Required unless the radiation is supplied by column.
    tz : str, float, datetime.tzinfo, or None, optional
        Time zone the timestamps are in: a fixed UTC offset in hours
        (e.g. ``-7``), an IANA zone name (e.g. ``"America/Denver"``), or a
        ``tzinfo``. Omit it when the index is already time-zone aware.
    sw_in_pot : str or None, default "SW_IN_POT"
        Column of precomputed potential radiation [W m⁻²] to use instead of
        the solar geometry, matched case-insensitively; the AmeriFlux BASE
        name by default. Pass None to always compute the geometry.

    Returns
    -------
    pandas.Series
        Boolean Series named ``"daytime"`` on the index of `df`, True where
        the Sun is above the horizon.

    Raises
    ------
    TypeError
        If `df` is not a DataFrame or DatetimeIndex, or `tz` is of an
        unusable type.
    ValueError
        If the geolocation is needed but missing or out of range; if the index
        holds a NaT, or is naive with no `tz` (or aware with one); or if the
        `sw_in_pot` column has gaps that cannot be filled because no
        geolocation was given.

    See Also
    --------
    potential_radiation : The radiation the threshold is applied to.
    monthly_climatologies : Uses this split to build the paired climatologies.

    Notes
    -----
    A precomputed column is preferred where present, so that the split matches
    whatever the data provider used; gaps in it are filled from the geometry
    when a geolocation is available, since a boolean flag has no room for a
    missing value.

    AmeriFlux BASE timestamps are in local standard time throughout the year.
    Pass the site's standard-time offset as a number, not an IANA zone name,
    or every summer sunrise moves an hour.

    Examples
    --------
    >>> import pandas as pd
    >>> times = pd.date_range("2020-12-21", periods=4, freq="6h")
    >>> partition_daynight(times, 40.0, -111.0, tz=-7).tolist()
    [False, False, True, False]

    A precomputed column wins over the geometry, and needs no geolocation:

    >>> df = pd.DataFrame({"SW_IN_POT": [0.0, 0.0, 550.0, 0.0]}, index=times)
    >>> partition_daynight(df).tolist()
    [False, False, True, False]
    """
    index = _timestamps(df)
    column = _find_column(df, sw_in_pot) if sw_in_pot else None

    if column is None:
        radiation = potential_radiation(df, latitude, longitude, tz)
    else:
        radiation = pd.to_numeric(df[column], errors="coerce").astype(float)
        missing = radiation.isna()
        if missing.any():
            if latitude is None or longitude is None:
                raise ValueError(
                    f"The '{column}' column has {int(missing.sum())} missing "
                    f"values, and a day-night flag has no room for one. Pass "
                    f"latitude and longitude so the gaps can be filled from "
                    f"the solar geometry, or drop those rows."
                )
            filled = potential_radiation(index[missing], latitude, longitude, tz)
            radiation = radiation.copy()
            radiation.loc[missing] = filled.to_numpy()

    daytime = np.asarray(radiation, dtype=float) > 0.0
    return pd.Series(daytime, index=index, name=DAYTIME)


def _resolve_footprints(model_or_ds: Any) -> tuple[xr.DataArray, pd.DataFrame | None]:
    """
    Find the time-resolved footprints, and the frame they were computed from.

    Parameters
    ----------
    model_or_ds : BaseFootprintModel, xarray.Dataset, or xarray.DataArray
        A run footprint model, its results Dataset, or the time-resolved
        footprint array itself.

    Returns
    -------
    f_2d : xarray.DataArray
        Time-resolved footprints, transposed to dims ``(time, x, y)``.
    df : pandas.DataFrame or None
        The model's input frame where one is available, so that a precomputed
        potential-radiation column can be picked up from it.

    Raises
    ------
    TypeError
        If `model_or_ds` is none of the accepted types.
    ValueError
        If the model has not been run, if a Dataset carries no recognisable
        time-resolved footprint variable, or if the array's dims are not
        ``(time, x, y)``.
    """
    source: pd.DataFrame | None = None

    if isinstance(model_or_ds, xr.DataArray):
        f_2d = model_or_ds
    elif isinstance(model_or_ds, xr.Dataset):
        found = [
            name
            for name in _FOOTPRINT_VARIABLES
            if name in model_or_ds and "time" in model_or_ds[name].dims
        ]
        if not found:
            raise ValueError(
                f"The Dataset carries no time-resolved footprint variable. "
                f"Looked for {', '.join(_FOOTPRINT_VARIABLES)} with a 'time' "
                f"dimension; it holds {', '.join(map(str, model_or_ds.data_vars))}."
            )
        f_2d = model_or_ds[found[0]]
    elif hasattr(model_or_ds, "f_2d"):
        f_2d = model_or_ds.f_2d
        if f_2d is None:
            raise ValueError(
                f"{type(model_or_ds).__name__}.f_2d is None: the model has "
                f"not been run, or it does not retain the time-resolved "
                f"footprints. Call run() first."
            )
        candidate = getattr(model_or_ds, "df", None)
        if isinstance(candidate, pd.DataFrame):
            source = candidate
    else:
        raise TypeError(
            f"Expected a footprint model, an xarray.Dataset of results, or a "
            f"time-resolved xarray.DataArray, got {type(model_or_ds).__name__}."
        )

    if not isinstance(f_2d, xr.DataArray):
        raise TypeError(
            f"The time-resolved footprints must be an xarray.DataArray, got "
            f"{type(f_2d).__name__}."
        )
    expected = {"time", "x", "y"}
    if set(f_2d.dims) != expected:
        raise ValueError(
            f"The time-resolved footprints must have dims (time, x, y), got "
            f"{f_2d.dims}."
        )
    return f_2d.transpose("time", "x", "y"), source


def _daytime_mask(
    times: pd.DatetimeIndex,
    daytime: pd.Series | np.ndarray | xr.DataArray | None,
    source: pd.DataFrame | None,
    latitude: float | None,
    longitude: float | None,
    tz: str | float | dt.tzinfo | None,
    sw_in_pot: str | None,
) -> np.ndarray:
    """
    Resolve the day-night split for the timestamps of a footprint series.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Footprint timestamps, from the ``time`` coordinate.
    daytime : pandas.Series, numpy.ndarray, xarray.DataArray, or None
        A precomputed flag. A Series is aligned on its index; an array is
        taken in order and must match the number of timestamps.
    source : pandas.DataFrame or None
        The model's input frame, searched for a `sw_in_pot` column.
    latitude, longitude : float or None
        Site geolocation [degrees], passed to :func:`partition_daynight`.
    tz : str, float, datetime.tzinfo, or None
        Time zone of `times`.
    sw_in_pot : str or None
        Precomputed potential-radiation column to prefer.

    Returns
    -------
    numpy.ndarray
        Boolean array, one flag per timestamp.

    Raises
    ------
    ValueError
        If a supplied `daytime` does not cover every timestamp or is the
        wrong length, or if the split cannot be computed.
    """
    if daytime is not None:
        if isinstance(daytime, xr.DataArray):
            daytime = daytime.to_series() if "time" in daytime.dims else daytime.values
        if isinstance(daytime, pd.Series):
            aligned = daytime.reindex(times)
            if aligned.isna().any():
                raise ValueError(
                    f"daytime does not cover every footprint timestamp: "
                    f"{int(aligned.isna().sum())} of {len(times)} are missing."
                )
            return np.asarray(aligned, dtype=bool)
        values = np.asarray(daytime)
        if values.shape != (len(times),):
            raise ValueError(
                f"daytime holds {values.shape} flags for {len(times)} "
                f"footprint timestamps."
            )
        return values.astype(bool)

    frame = pd.DataFrame(index=times)
    column = _find_column(source, sw_in_pot) if sw_in_pot else None
    if column is not None and isinstance(source.index, pd.DatetimeIndex):
        frame[column] = source[column].reindex(times)
    return np.asarray(
        partition_daynight(frame, latitude, longitude, tz, sw_in_pot), dtype=bool
    )


def monthly_climatologies(
    model_or_ds: Any,
    partition: str = "month+daynight",
    latitude: float | None = None,
    longitude: float | None = None,
    tz: str | float | dt.tzinfo | None = None,
    daytime: pd.Series | np.ndarray | xr.DataArray | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    dx: float | None = None,
    dy: float | None = None,
    min_times: int = 1,
    sw_in_pot: str | None = SW_IN_POT_COLUMN,
) -> xr.Dataset:
    """
    Aggregate time-resolved footprints into monthly climatologies, Sect. 2.2.

    Chu et al. (2021) aggregate every half-hourly footprint of a month into a
    daytime and a nighttime climatology, truncate each at the 80 % contour of
    source weights, and rescale the retained cells to sum to one. The result
    is the unit of analysis for everything downstream: the metrics of Eqs. 1-3
    and the footprint-weighted statistics of Eqs. 5-6.

    Parameters
    ----------
    model_or_ds : BaseFootprintModel, xarray.Dataset, or xarray.DataArray
        A run footprint model, its results Dataset, or the time-resolved
        footprints themselves with dims ``(time, x, y)``. A model is also
        searched for a `sw_in_pot` column in its input frame.
    partition : {"month+daynight", "month"}, default "month+daynight"
        How to group the timesteps. ``"month+daynight"`` splits each month
        into daytime and nighttime as in the paper; ``"month"`` aggregates
        every hour of the month into one climatology, labelled ``"all"``.
    latitude : float, optional
        Site latitude, positive north [degrees].
    longitude : float, optional
        Site longitude, positive east [degrees]; western longitudes are
        negative.
    tz : str, float, datetime.tzinfo, or None, optional
        Time zone the footprint timestamps are in, as accepted by
        :func:`partition_daynight`.
    daytime : pandas.Series, numpy.ndarray, or xarray.DataArray, optional
        A precomputed day-night flag, used instead of computing one. A Series
        is aligned on its index and must cover every timestamp; an array is
        taken in order.
    fraction : float, default 0.8
        Source-weight fraction each climatology is truncated at.
    dx, dy : float, optional
        Grid spacing [m]. Inferred from the ``x`` and ``y`` coordinates when
        omitted.
    min_times : int, default 1
        Fewest contributing timesteps a group needs before it is aggregated.
        Thinner groups come back as all-NaN weights with their ``n_times``
        recorded, rather than as a climatology built from a handful of
        half-hours.
    sw_in_pot : str or None, default "SW_IN_POT"
        Precomputed potential-radiation column to prefer over the solar
        geometry, matched case-insensitively.

    Returns
    -------
    xarray.Dataset
        Dims ``(month, period, x, y)``, carrying

        ``footprint_climatology``
            Truncated, renormalised weights [-] summing to one over ``(x, y)``
            in every aggregated group, and all-NaN in the groups left empty or
            held back by `min_times`.
        ``n_times``
            Timesteps that contributed positive weight to each group.
        ``contour_level``
            Source-weight density [m⁻²] at the truncation contour. It varies
            from group to group, so it lives here rather than in the stacked
            array's attributes, and :func:`climatology_metrics` reports it as
            ``nan`` for a slice; read it from here alongside.
        ``contour_n_cells``
            Cells retained inside the contour.

        The ``month`` coordinate is the first instant of each calendar month
        present, and ``period`` is ``["daytime", "nighttime"]`` or ``["all"]``.

    Raises
    ------
    TypeError
        If `model_or_ds` is of an unusable type.
    ValueError
        If `partition` is not one of the two accepted values; if `min_times`
        is below one; if the footprints carry no usable ``time`` coordinate or
        their dims are not ``(time, x, y)``; or if the day-night split cannot
        be resolved.

    See Also
    --------
    truncate_to_contour : The per-group truncation and renormalisation.
    seasonal_overlap : Eq. 2, taken over this Dataset's ``month`` dimension.
    daynight_overlap : Eq. 3, over its two ``period`` slices.
    climatology_metrics : Fetch, area, and symmetry of one group.

    Notes
    -----
    Months are calendar months of a specific year, not months of the year, so
    a multi-year record yields a separate January per year rather than one
    blended January. The paper's indices are site-*year* properties; select a
    year before passing the ``month`` dimension to :func:`seasonal_overlap`.

    Each group is summed over its contributing timesteps and divided by their
    count, matching the climatology convention of the footprint models. The
    divisor cancels in the renormalisation, so it affects only the recorded
    ``contour_level``, which stays comparable across groups because of it.

    A group with no data -- polar night, or a month the tower was down -- is
    all-NaN rather than all-zero, so that it cannot be mistaken for a
    footprint that legitimately puts no weight anywhere. Drop those before
    the overlap indices, which require every month to sum to one:
    ``ds.where(ds.n_times > 0, drop=True)``.

    Examples
    --------
    >>> from fluxfootprints import build_climatology, monthly_climatologies
    >>> model = build_climatology(df)                        # doctest: +SKIP
    >>> clim = monthly_climatologies(                        # doctest: +SKIP
    ...     model, latitude=40.1, longitude=-111.9, tz=-7
    ... )
    >>> clim.footprint_climatology.dims                      # doctest: +SKIP
    ('month', 'period', 'x', 'y')
    """
    if partition not in _PARTITIONS:
        raise ValueError(
            f"partition must be one of {_PARTITIONS}, got {partition!r}."
        )
    if int(min_times) < 1:
        raise ValueError(f"min_times must be at least 1, got {min_times!r}.")
    min_times = int(min_times)

    f_2d, source = _resolve_footprints(model_or_ds)
    if "time" not in f_2d.coords:
        raise ValueError(
            "The time-resolved footprints carry no 'time' coordinate, so they "
            "cannot be grouped into months."
        )
    times = pd.DatetimeIndex(f_2d["time"].values)
    if times.hasnans:
        raise ValueError(
            "The 'time' coordinate holds NaT, so those footprints belong to no "
            "month. Drop them first."
        )

    if partition == "month+daynight":
        is_daytime = _daytime_mask(
            times, daytime, source, latitude, longitude, tz, sw_in_pot
        )
        periods = [DAYTIME, NIGHTTIME]
        selectors = [is_daytime, ~is_daytime]
    else:
        periods = [ALL_HOURS]
        selectors = [np.ones(len(times), dtype=bool)]

    month_of = times.values.astype("datetime64[M]")
    months = np.unique(month_of)

    shape = (len(months), len(periods), f_2d.sizes["x"], f_2d.sizes["y"])
    weights = np.full(shape, np.nan)
    n_times = np.zeros(shape[:2], dtype="int64")
    levels = np.full(shape[:2], np.nan)
    n_cells = np.zeros(shape[:2], dtype="int64")

    per_timestep = f_2d.sum(dim=("x", "y"))
    contributes = np.asarray(per_timestep.values, dtype=float) > 0.0

    for m, month in enumerate(months):
        in_month = month_of == month
        for p, selector in enumerate(selectors):
            chosen = np.flatnonzero(in_month & selector & contributes)
            n_times[m, p] = chosen.size
            if chosen.size < min_times:
                continue

            climatology = f_2d.isel(time=chosen).sum(dim="time") / chosen.size
            truncated = truncate_to_contour(
                climatology, dx, dy, fraction=fraction, renormalize=True
            )
            weights[m, p] = truncated.values
            levels[m, p] = float(truncated.attrs["contour_level"])
            n_cells[m, p] = int(truncated.attrs["contour_n_cells"])

    coords: dict[str, Any] = {
        _MONTH_DIM: months.astype("datetime64[ns]"),
        _PERIOD_DIM: periods,
    }
    for axis in ("x", "y"):
        if axis in f_2d.coords:
            coords[axis] = f_2d.coords[axis]

    dims = (_MONTH_DIM, _PERIOD_DIM, "x", "y")
    group_dims = (_MONTH_DIM, _PERIOD_DIM)
    ds = xr.Dataset(
        {
            "footprint_climatology": (
                dims,
                weights,
                {
                    "units": "1",
                    "long_name": "renormalised footprint climatology",
                    "contour_fraction": float(fraction),
                },
            ),
            "n_times": (
                group_dims,
                n_times,
                {"long_name": "contributing timesteps"},
            ),
            "contour_level": (
                group_dims,
                levels,
                {"units": "m-2", "long_name": "source weight at the contour"},
            ),
            "contour_n_cells": (
                group_dims,
                n_cells,
                {"long_name": "cells inside the contour"},
            ),
        },
        coords=coords,
    )
    ds.attrs.update(
        {
            "partition": partition,
            "contour_fraction": float(fraction),
            "min_times": min_times,
            "n_timesteps": len(times),
            "description": (
                "Monthly footprint climatologies truncated at the "
                f"{fraction:.0%} source-weight contour and renormalised to "
                "unit sum, after Chu et al. (2021), Sect. 2.2."
            ),
            "reference": (
                "Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350."
            ),
        }
    )
    return ds


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

    ``contour_level`` is ``nan`` too where a truncated input has lost that
    attribute, as a climatology sliced out of the stack from
    :func:`monthly_climatologies` has: the level varies from group to group,
    so it is a data variable of that Dataset rather than an attribute of the
    stacked array. Read it from ``ds.contour_level`` alongside.

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
        level = float(fclim.attrs.get("contour_level", float("nan")))
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


def _retained_weight(
    weights: np.ndarray,
    valid: np.ndarray,
    name: str,
) -> tuple[float, float]:
    """
    Split a footprint's weight into the part that fell on cells holding data.

    Parameters
    ----------
    weights : numpy.ndarray
        Validated, non-negative source weights.
    valid : numpy.ndarray
        Boolean array of the same shape, True where the field holds data.
    name : str
        How to refer to the weights in the error message.

    Returns
    -------
    kept : float
        Weight lying on the valid cells, the denominator of the renormalised
        weighted mean.
    retained : float
        `kept` as a fraction of the footprint's total weight, in [0, 1].

    Raises
    ------
    ValueError
        If the footprint carries no weight at all, leaving nothing to
        renormalise and no fraction to report.
    """
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            f"{name} carries no source weight -- its cells sum to "
            f"{total:.6g} -- so there is nothing to average under."
        )
    kept = float(weights[valid].sum())
    return kept, kept / total


def _class_codes(codes: np.ndarray) -> np.ndarray:
    """
    Return class codes as integers when every one of them is integral.

    Parameters
    ----------
    codes : numpy.ndarray
        Unique class codes, as floats after the raster alignment path.

    Returns
    -------
    numpy.ndarray
        `codes` as int64 if they are all whole numbers, unchanged otherwise.

    Notes
    -----
    :func:`sample_raster_on_grid` returns float64 whatever the source dtype,
    so an NLCD code arrives as ``41.0``; indexing a composition by ``41`` is
    what a caller holding a class lookup table will try.
    """
    values = np.asarray(codes, dtype=float)
    if np.all(values == np.rint(values)):
        return values.astype(np.int64)
    return values


def _composition(
    codes: np.ndarray,
    weights: np.ndarray | None,
    name: str,
    retained_weight: float,
) -> pd.Series:
    """
    Reduce the class codes of the contributing cells to a fraction per class.

    Parameters
    ----------
    codes : numpy.ndarray
        Class code of every contributing cell, flattened, already restricted
        to the cells holding both a code and weight.
    weights : numpy.ndarray or None
        Weight of each cell in `codes`, or None for the uniform weights of an
        unweighted composition.
    name : str
        Name of the returned series.
    retained_weight : float
        Fraction of the weight the cells in `codes` carry, recorded in attrs.

    Returns
    -------
    pandas.Series
        Fraction per class, indexed by class code in ascending order and
        summing to 1; empty when `codes` is.
    """
    unique, inverse = np.unique(codes, return_inverse=True)
    totals = np.bincount(
        np.ravel(inverse), weights=weights, minlength=unique.size
    ).astype(float)
    total = float(totals.sum())
    fractions = totals / total if total > 0.0 else totals
    series = pd.Series(
        fractions,
        index=pd.Index(_class_codes(unique), name="class"),
        name=name,
        dtype=float,
    )
    series.attrs["retained_weight"] = float(retained_weight)
    series.attrs["n_cells"] = int(codes.size)
    return series


def footprint_weighted_value(
    weights: xr.DataArray | np.ndarray,
    raster: xr.DataArray | np.ndarray,
) -> WeightedValue:
    """
    Compute the footprint-weighted value of a continuous field, Eq. 5.

    .. math:: EVI_{footprint} = \\sum_{j=1}^{J} \\varphi_j EVI_j

    Parameters
    ----------
    weights : xarray.DataArray or numpy.ndarray
        Footprint source weights with dims ``(x, y)``, normally the truncated,
        renormalised climatology from :func:`truncate_to_contour`. Weights that
        sum to something other than 1 are accepted and renormalised; the
        retained fraction is then reported relative to their own total.
    raster : xarray.DataArray or numpy.ndarray
        Continuous field on the same grid, e.g. Landsat EVI, NaN where it holds
        no data. :func:`sample_raster_on_grid` delivers an external product in
        exactly that form. DataArrays are compared cell for cell rather than
        aligned on their coordinates, so a mismatched grid raises instead of
        silently joining.

    Returns
    -------
    WeightedValue
        The weighted value, the fraction of the footprint weight that fell on
        cells holding data, and the number of contributing cells. ``value`` is
        ``nan`` and ``retained_weight`` 0 when the raster covers no weighted
        cell.

    Raises
    ------
    ValueError
        If the two arrays are not on the same grid; if `weights` is empty or
        holds a non-finite or negative weight; or if it carries no weight at
        all.

    See Also
    --------
    target_area_value : The unweighted counterpart over a target disc.
    sensor_location_bias : Compares the two, Eq. 6.
    footprint_weighted_composition : The categorical counterpart.

    Notes
    -----
    Cells where `raster` is NaN are dropped and the surviving weights
    renormalised, so partial raster coverage biases the result toward the
    covered part of the source area rather than returning NaN outright.
    ``retained_weight`` is what says how far that went: a value carrying 0.6
    of the footprint weight is a different measurement from one carrying all
    of it, and Chu et al. (2021) kept only scenes covering the source area.

    Examples
    --------
    >>> evi = sample_raster_on_grid("evi.tif", model.x, model.y, lat, lon)  # doctest: +SKIP
    >>> result = footprint_weighted_value(truncate_to_contour(fclim), evi)  # doctest: +SKIP
    >>> result.value, result.retained_weight                                # doctest: +SKIP
    (0.42, 1.0)
    """
    _check_same_grid(weights, raster, "weights", "raster")
    phi = _weight_values(weights, "weights")
    values = np.asarray(raster, dtype=float)

    valid = np.isfinite(values)
    kept, retained = _retained_weight(phi, valid, "weights")
    contributing = valid & (phi > 0.0)
    if kept <= 0.0:
        return WeightedValue(float("nan"), 0.0, 0)

    value = float((phi[contributing] * values[contributing]).sum() / kept)
    return WeightedValue(value, retained, int(np.count_nonzero(contributing)))


def footprint_weighted_composition(
    weights: xr.DataArray | np.ndarray,
    landcover: xr.DataArray | np.ndarray,
) -> pd.Series:
    """
    Compute the footprint-weighted composition of a categorical field.

    The categorical counterpart of :func:`footprint_weighted_value`: every
    class takes the share of the footprint weight that falls on it, which is
    P_footprint of Sect. 2.4 once multiplied by 100.

    Parameters
    ----------
    weights : xarray.DataArray or numpy.ndarray
        Footprint source weights with dims ``(x, y)``, as for
        :func:`footprint_weighted_value`.
    landcover : xarray.DataArray or numpy.ndarray
        Categorical raster on the same grid, typically the integer codes of
        the consolidated NLCD / Land Cover of Canada groups of Table S6, NaN
        where it holds no data.

    Returns
    -------
    pandas.Series
        Weight fraction per class, indexed by class code in ascending order
        (index name ``class``) and summing to 1. Empty when no cell carries
        both weight and a class code. ``Series.attrs`` holds
        ``retained_weight`` and ``n_cells``, as on :class:`WeightedValue`.

    Raises
    ------
    ValueError
        As for :func:`footprint_weighted_value`.

    See Also
    --------
    target_area_composition : The unweighted counterpart over a target disc.

    Notes
    -----
    Integral class codes come back as integers whatever the raster's dtype,
    since the alignment path returns float64. The fractions are shares of the
    retained weight, so this composition and a target-area one over the same
    product are directly comparable class by class -- the comparison
    :func:`classify_categorical` rests on.

    Examples
    --------
    >>> nlcd = sample_raster_on_grid(                       # doctest: +SKIP
    ...     "nlcd.tif", model.x, model.y, lat, lon, categorical=True
    ... )
    >>> footprint_weighted_composition(weights, nlcd)       # doctest: +SKIP
    class
    41    0.62
    81    0.38
    Name: footprint_fraction, dtype: float64
    """
    _check_same_grid(weights, landcover, "weights", "landcover")
    phi = _weight_values(weights, "weights")
    codes = np.asarray(landcover, dtype=float)

    valid = np.isfinite(codes)
    _, retained = _retained_weight(phi, valid, "weights")
    contributing = valid & (phi > 0.0)
    return _composition(
        codes[contributing],
        phi[contributing],
        "footprint_fraction",
        retained,
    )


def target_area_value(
    raster: xr.DataArray | np.ndarray,
    x: np.ndarray | xr.DataArray,
    y: np.ndarray | xr.DataArray,
    radius: float,
) -> WeightedValue:
    """
    Compute the unweighted mean of a continuous field over a target area.

    EVI_target of Sect. 2.4: the field averaged over the disc of radius
    `radius` around the tower, every cell counting the same, against which the
    footprint-weighted value of Eq. 5 is compared.

    Parameters
    ----------
    raster : xarray.DataArray or numpy.ndarray
        Continuous field with dims ``(x, y)`` on the tower-centred grid, NaN
        where it holds no data, as :func:`sample_raster_on_grid` returns it.
    x, y : numpy.ndarray or xarray.DataArray
        Cell-centre offsets from the tower [m] (``model.x``, ``model.y``),
        which is the grid `raster` must be on.
    radius : float
        Target-area radius [m], e.g. one of :data:`TARGET_RADII`.

    Returns
    -------
    WeightedValue
        The arithmetic mean over the disc, the fraction of its cells that held
        data, and the number of those cells. ``value`` is ``nan`` and
        ``retained_weight`` 0 when no cell inside the disc holds data.

    Raises
    ------
    ValueError
        If `radius` is not positive and finite; if `raster` is not on the grid
        `x` and `y` describe; or if the disc is smaller than one grid cell, so
        that no cell centre falls inside it.

    See Also
    --------
    footprint_weighted_value : The footprint-weighted counterpart, Eq. 5.
    target_area_mask : The disc this averages over.

    Notes
    -----
    A disc reaching past the edge of the domain is silently clipped to it, as
    :func:`target_area_mask` documents, and a raster not covering the whole
    disc lowers ``retained_weight`` rather than the value going NaN. The two
    together are what make a target-area value at 3000 m trustworthy or not,
    and the paper's larger radii are exactly where domains and scenes run out.
    """
    mask = target_area_mask(x, y, radius)
    _check_same_grid(mask, raster, "the target area", "raster")

    inside = np.asarray(mask.values)
    n_inside = int(np.count_nonzero(inside))
    if n_inside == 0:
        raise ValueError(
            f"No cell centre lies within {float(radius):g} m of the tower, so "
            f"the target area holds no cells. Use a radius of at least half "
            f"the grid spacing, or a finer grid."
        )

    values = np.asarray(raster, dtype=float)
    contributing = inside & np.isfinite(values)
    n_cells = int(np.count_nonzero(contributing))
    if n_cells == 0:
        return WeightedValue(float("nan"), 0.0, 0)

    return WeightedValue(
        float(values[contributing].mean()),
        n_cells / n_inside,
        n_cells,
    )


def target_area_composition(
    landcover: xr.DataArray | np.ndarray,
    x: np.ndarray | xr.DataArray,
    y: np.ndarray | xr.DataArray,
    radius: float,
) -> pd.Series:
    """
    Compute the composition of a categorical field over a target area.

    The categorical counterpart of :func:`target_area_value`: every class takes
    the share of the disc's cells it covers, which is P_target of Sect. 2.4
    once multiplied by 100.

    Parameters
    ----------
    landcover : xarray.DataArray or numpy.ndarray
        Categorical raster with dims ``(x, y)`` on the tower-centred grid, NaN
        where it holds no data.
    x, y : numpy.ndarray or xarray.DataArray
        Cell-centre offsets from the tower [m] (``model.x``, ``model.y``).
    radius : float
        Target-area radius [m].

    Returns
    -------
    pandas.Series
        Area fraction per class, indexed by class code in ascending order
        (index name ``class``) and summing to 1. Empty when no cell inside the
        disc holds a class code. ``Series.attrs`` holds ``retained_weight``,
        the fraction of the disc's cells that held one, and ``n_cells``.

    Raises
    ------
    ValueError
        As for :func:`target_area_value`.

    See Also
    --------
    footprint_weighted_composition : The footprint-weighted counterpart.

    Notes
    -----
    Cells without a class code are dropped from numerator and denominator
    alike, so the fractions describe the classified part of the disc; how much
    of it that was is ``attrs["retained_weight"]``.
    """
    mask = target_area_mask(x, y, radius)
    _check_same_grid(mask, landcover, "the target area", "landcover")

    inside = np.asarray(mask.values)
    n_inside = int(np.count_nonzero(inside))
    if n_inside == 0:
        raise ValueError(
            f"No cell centre lies within {float(radius):g} m of the tower, so "
            f"the target area holds no cells. Use a radius of at least half "
            f"the grid spacing, or a finer grid."
        )

    codes = np.asarray(landcover, dtype=float)
    contributing = inside & np.isfinite(codes)
    return _composition(
        codes[contributing],
        None,
        "target_fraction",
        int(np.count_nonzero(contributing)) / n_inside,
    )


# ------------------------------
# Bias and regression (Sect. 2.4)
# ------------------------------


def _relative_bias(footprint_value: float, target_value: float) -> float:
    """
    Evaluate the sensor location bias of Eq. 6 for one pair of values.

    .. math:: \\Delta = \\frac{EVI_{footprint} - EVI_{target}}{EVI_{target}}

    Parameters
    ----------
    footprint_value : float
        Footprint-weighted value, EVI_footprint of Eq. 5.
    target_value : float
        Target-area mean, EVI_target.

    Returns
    -------
    float
        The relative bias as a fraction, or ``nan`` when either value is
        non-finite or the target-area mean is zero, which leaves the ratio
        undefined rather than infinite.
    """
    footprint = float(footprint_value)
    target = float(target_value)
    if not np.isfinite(footprint) or not np.isfinite(target) or target == 0.0:
        return float("nan")
    return (footprint - target) / target


def _within_threshold(
    delta: np.ndarray,
    threshold: float = BIAS_THRESHOLD,
) -> pd.arrays.BooleanArray:
    """
    Flag the biases meeting the paper's threshold, keeping the gaps missing.

    Parameters
    ----------
    delta : numpy.ndarray
        Sensor location biases as fractions, possibly holding ``nan``.
    threshold : float, default BIAS_THRESHOLD
        ``|delta|`` at or below which a period counts as representative.
        Chu et al. (2021) adopt 0.10, following Chen et al. (2011) and Kim
        et al. (2006); 0.05 also appears in that literature.

    Returns
    -------
    pandas.arrays.BooleanArray
        True where ``|delta| <= threshold``, and ``pd.NA`` where `delta` is
        not finite.

    Notes
    -----
    The missing entries are what keep a period whose bias could not be
    computed -- a scene not covering the target area -- from counting as a
    failure when the flags are averaged into the percentages of Fig. 7.
    """
    values = np.asarray(delta, dtype=float)
    within = pd.array(np.abs(values) <= float(threshold), dtype="boolean")
    within[~np.isfinite(values)] = pd.NA
    return within


def sensor_location_bias(
    w: xr.DataArray | np.ndarray,
    raster: xr.DataArray | np.ndarray,
    x: np.ndarray | xr.DataArray,
    y: np.ndarray | xr.DataArray,
    radii: Sequence[float] = TARGET_RADII,
) -> pd.DataFrame:
    """
    Compute the sensor location bias Delta against each target area, Eq. 6.

    .. math:: \\Delta = \\frac{EVI_{footprint} - EVI_{target}}{EVI_{target}}

    After Schmid and Lloyd (1999); the time-explicit footprint-to-target-area
    bias for one period, evaluated against the series of target radii of
    Sect. 2.1. The footprint-weighted value of Eq. 5 does not depend on the
    radius, so it is computed once and repeated down the frame; only the
    target-area mean moves.

    Parameters
    ----------
    w : xarray.DataArray or numpy.ndarray
        Footprint source weights with dims ``(x, y)``, normally one period's
        truncated, renormalised climatology from :func:`truncate_to_contour`
        or a slice of :func:`monthly_climatologies`. Weights summing to
        something other than 1 are renormalised, as in
        :func:`footprint_weighted_value`.
    raster : xarray.DataArray or numpy.ndarray
        Continuous field on the same grid, e.g. the Landsat EVI scene matched
        to this period, NaN where it holds no data.
    x, y : numpy.ndarray or xarray.DataArray
        Cell-centre offsets from the tower [m] (``model.x``, ``model.y``),
        which is the grid both `w` and `raster` must be on.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m], evaluated and reported in the order given.

    Returns
    -------
    pandas.DataFrame
        One row per radius, with columns

        ``radius``
            Target-area radius [m].
        ``value_footprint``
            Footprint-weighted value, Eq. 5. Constant down the frame.
        ``value_target``
            Target-area mean over the disc of that radius.
        ``delta``
            Sensor location bias as a fraction, not a percentage; multiply by
            100 to match the paper's figures. Positive values mean the
            footprint covered higher values than its surroundings, which held
            at every target radius in the paper. ``nan`` where either value is
            non-finite or the target-area mean is zero.
        ``within_threshold``
            Whether ``|delta| <=`` :data:`BIAS_THRESHOLD`, in pandas' nullable
            boolean dtype, and ``pd.NA`` where `delta` is undefined.

        ``DataFrame.attrs["bias_threshold"]`` records the threshold applied.

    Raises
    ------
    ValueError
        If `radii` is empty; if any radius is not positive and finite, or is
        smaller than half the grid spacing so that its disc holds no cell
        centre; if `w`, `raster`, and the grid `x` and `y` describe are not one
        and the same; or if `w` holds a non-finite or negative weight, or no
        weight at all.

    See Also
    --------
    sensor_location_bias_series : Maps this over a time-indexed collection.
    footprint_weighted_value : EVI_footprint, Eq. 5.
    target_area_value : EVI_target.
    model2_regression : The site-level counterpart, Eq. 7.

    Notes
    -----
    A raster covering only part of the footprint or the disc lowers the
    retained weight of the underlying averages rather than making them NaN, so
    a `delta` here can rest on partial coverage. Call
    :func:`footprint_weighted_value` and :func:`target_area_value` directly
    when that fraction matters; Chu et al. (2021) kept only scenes that were
    cloud-free within 3000 m of the tower.

    Examples
    --------
    >>> evi = sample_raster_on_grid("evi.tif", model.x, model.y, lat, lon)  # doctest: +SKIP
    >>> bias = sensor_location_bias(                                       # doctest: +SKIP
    ...     truncate_to_contour(model.fclim_2d), evi, model.x, model.y
    ... )
    >>> bias[["radius", "delta", "within_threshold"]]                      # doctest: +SKIP
       radius     delta  within_threshold
    0   250.0  0.021...              True
    1   500.0  0.064...              True
    ...

    References
    ----------
    Schmid, H. P., and Lloyd, C. R. (1999). Spatial representativeness and the
    location bias of flux footprints over inhomogeneous areas. *Agric. For.
    Meteorol.*, **93**, 195-209.

    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """
    radii_values = [float(radius) for radius in radii]
    if not radii_values:
        raise ValueError(
            "radii holds no target areas, so there is nothing to compare the "
            "footprint against. Pass at least one radius, e.g. TARGET_RADII."
        )

    footprint = footprint_weighted_value(w, raster)
    rows = [
        {
            "radius": radius,
            "value_footprint": footprint.value,
            "value_target": target.value,
            "delta": _relative_bias(footprint.value, target.value),
        }
        for radius, target in (
            (radius, target_area_value(raster, x, y, radius)) for radius in radii_values
        )
    ]

    frame = pd.DataFrame(
        rows, columns=["radius", "value_footprint", "value_target", "delta"]
    ).astype(float)
    frame["within_threshold"] = _within_threshold(frame["delta"].to_numpy())
    frame.attrs["bias_threshold"] = BIAS_THRESHOLD
    return frame


def _bias_series_items(
    pairs: Mapping[Any, Any] | pd.Series | Sequence[Any],
) -> list[tuple[Any, Any, Any]]:
    """
    Normalise the accepted period collections into ``(time, w, raster)`` rows.

    Parameters
    ----------
    pairs : mapping, pandas.Series, or sequence
        The collection :func:`sensor_location_bias_series` was handed.

    Returns
    -------
    list of tuple
        One ``(time, climatology, raster)`` triple per period, in the order
        the collection presented them.

    Raises
    ------
    TypeError
        If `pairs` is not iterable, or an entry is not the pair or triple the
        form it was passed in calls for.
    """
    if isinstance(pairs, (Mapping, pd.Series)):
        labelled = list(pairs.items())
    else:
        try:
            labelled = [(None, entry) for entry in pairs]
        except TypeError as exc:
            raise TypeError(
                f"pairs must be a mapping of time -> (climatology, raster), a "
                f"pandas Series of such pairs, or an iterable of "
                f"(time, climatology, raster) triples, got "
                f"{type(pairs).__name__}."
            ) from exc

    items: list[tuple[Any, Any, Any]] = []
    for label, entry in labelled:
        parts = entry if isinstance(entry, (tuple, list)) else None
        if label is None:
            if parts is not None and len(parts) == 3:
                items.append((parts[0], parts[1], parts[2]))
                continue
            hint = (
                " A bare (climatology, raster) pair carries no time label; "
                "pass a mapping keyed by time, or add the time as the first "
                "element."
                if parts is not None and len(parts) == 2
                else ""
            )
            raise TypeError(
                f"Every entry of an iterable of periods must be a "
                f"(time, climatology, raster) triple, got "
                f"{type(entry).__name__}.{hint}"
            )
        if parts is None or len(parts) != 2:
            raise TypeError(
                f"Every value of a mapping of periods must be a "
                f"(climatology, raster) pair, got {type(entry).__name__} at "
                f"time {label!r}."
            )
        items.append((label, parts[0], parts[1]))
    return items


def sensor_location_bias_series(
    pairs: Mapping[Any, Any] | pd.Series | Sequence[Any],
    x: np.ndarray | xr.DataArray,
    y: np.ndarray | xr.DataArray,
    radii: Sequence[float] = TARGET_RADII,
) -> pd.DataFrame:
    """
    Compute the sensor location bias of every period of a record, Eq. 6.

    :func:`sensor_location_bias` mapped over a time-indexed collection of
    matched climatology / field pairs and concatenated: the site-months of
    Chu et al. (2021), each a monthly footprint climatology paired with the
    Landsat scene retrieved within it. Grouping the result by ``radius`` and
    averaging ``within_threshold`` reproduces the percentages within the
    +/-10 % threshold of Sect. 3.3 and Fig. 7.

    Parameters
    ----------
    pairs : mapping, pandas.Series, or sequence
        The periods, in any of three forms:

        * a mapping of time label -> ``(climatology, raster)``;
        * a :class:`pandas.Series` indexed by time holding such pairs;
        * an iterable of ``(time, climatology, raster)`` triples.

        Each climatology and raster is what :func:`sensor_location_bias` takes
        as `w` and `raster`: a truncated, renormalised climatology and the
        field matched to that period, both on the grid `x` and `y` describe.
        Periods are evaluated in the order presented, not sorted.
    x, y : numpy.ndarray or xarray.DataArray
        Cell-centre offsets from the tower [m], shared by every period.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m].

    Returns
    -------
    pandas.DataFrame
        The frames of :func:`sensor_location_bias` stacked under a leading
        ``time`` column, over a fresh :class:`~pandas.RangeIndex`: columns
        ``time``, ``radius``, ``value_footprint``, ``value_target``, ``delta``,
        and ``within_threshold``, one row per period and radius.
        ``DataFrame.attrs["bias_threshold"]`` records the threshold applied.

    Raises
    ------
    TypeError
        If `pairs` is not one of the three accepted forms.
    ValueError
        If `pairs` holds no periods, or for any of the reasons
        :func:`sensor_location_bias` raises, with the offending time label
        prepended to the message.

    See Also
    --------
    sensor_location_bias : The single-period computation this maps.
    monthly_climatologies : Produces the per-period climatologies.
    evaluate_vegetation_index : Regresses the same pairs, Eq. 7.

    Notes
    -----
    The paper matched a scene to the climatology of the month it was retrieved
    in, and its Fig. 7 pools 3307 such site-months across 214 sites. Pairing is
    the caller's job here: this function evaluates the pairs it is given, in
    the order it is given them.

    Examples
    --------
    >>> pairs = {                                        # doctest: +SKIP
    ...     month: (clim.sel(month=month, period="daytime"), scenes[month])
    ...     for month in scenes
    ... }
    >>> bias = sensor_location_bias_series(pairs, model.x, model.y)  # doctest: +SKIP
    >>> bias.groupby("radius")["within_threshold"].mean()            # doctest: +SKIP
    radius
    250.0     0.73
    3000.0    0.42
    Name: within_threshold, dtype: Float64

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 3.3.
    """
    items = _bias_series_items(pairs)
    if not items:
        raise ValueError(
            "pairs holds no periods, so there is no series to compute. Pass "
            "at least one (climatology, raster) pair."
        )

    frames: list[pd.DataFrame] = []
    for label, weights, raster in items:
        try:
            frame = sensor_location_bias(weights, raster, x, y, radii=radii)
        except ValueError as exc:
            raise ValueError(f"At time {label!r}: {exc}") from exc
        frame.insert(0, "time", label)
        frames.append(frame)

    series = pd.concat(frames, ignore_index=True)
    series.attrs["bias_threshold"] = BIAS_THRESHOLD
    return series


def _finite_pairs(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop the pairs either variable is missing from.

    Parameters
    ----------
    x, y : array_like
        The two variables, of equal length.

    Returns
    -------
    x, y : numpy.ndarray
        The pairs where both values are finite, in their original order.

    Raises
    ------
    ValueError
        If the inputs differ in length.
    """
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.size != ys.size:
        raise ValueError(
            f"x and y must pair up one for one, got {xs.size} and {ys.size} "
            f"values. A regression of Eq. 7 runs over matched periods, so "
            f"drop the unmatched ones before fitting."
        )
    keep = np.isfinite(xs) & np.isfinite(ys)
    return xs[keep], ys[keep]


def _rma_slope_intercept(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the RMA slope and intercept, vectorised over resamples.

    Parameters
    ----------
    x, y : numpy.ndarray
        Finite values with the pairs along the last axis. A 2-D input is read
        as one resample per row.

    Returns
    -------
    slope, intercept : numpy.ndarray
        One estimate per row, ``nan`` where a row leaves the slope undefined:
        either variable constant, or a correlation of exactly zero, which
        fixes the magnitude of the RMA slope but not its sign.
    """
    mean_x = x.mean(axis=-1)
    mean_y = y.mean(axis=-1)
    dx = x - mean_x[..., None]
    dy = y - mean_y[..., None]
    # Population moments; the ddof cancels in every ratio taken below.
    sd_x = np.sqrt((dx**2).mean(axis=-1))
    sd_y = np.sqrt((dy**2).mean(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (dx * dy).mean(axis=-1) / (sd_x * sd_y)
        slope = np.sign(r) * sd_y / sd_x
    slope = np.where(np.isfinite(r) & (r != 0.0), slope, np.nan)
    return slope, mean_y - slope * mean_x


def rma_regression(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    ci_level: float = 0.95,
    ci_method: str = "analytical",
    n_boot: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> RMAFit:
    """
    Fit the reduced major axis (model II) regression of Eq. 7.

    .. math:: y \\sim \\beta_0 + \\beta_1 x

    The reduced major axis -- equivalently the standard major axis, the
    ``"SMA"`` row of R's ``lmodel2`` -- minimises the triangular areas between
    the points and the line rather than the vertical residuals alone. Its
    slope is the ordinary least squares slope divided by the Pearson
    correlation coefficient, i.e. ``sign(r) * sd(y) / sd(x)``.

    Parameters
    ----------
    x : array_like
        Predictor. In Eq. 7 the footprint-weighted values of Eq. 5.
    y : array_like
        Response, of the same length as `x`. In Eq. 7 the target-area means.
    ci_level : float, default 0.95
        Confidence level of the reported intervals, strictly between 0 and 1.
    ci_method : {'analytical', 'bootstrap'}, default 'analytical'
        How to bracket the slope. ``'analytical'`` uses the closed form of
        McArdle (1988) that ``lmodel2`` reports; ``'bootstrap'`` takes
        percentile limits over `n_boot` paired resamples, which does not
        assume bivariate normality but needs a decent `n`.
    n_boot : int, default 2000
        Number of resamples, used only when ``ci_method='bootstrap'``.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator for the bootstrap, for reproducible limits.

    Returns
    -------
    RMAFit
        The slope, intercept, R-squared, p-value, and the two confidence
        intervals.

    Raises
    ------
    ValueError
        If the inputs differ in length; if fewer than three finite pairs
        remain, leaving no residual degrees of freedom; if `ci_level` is not
        strictly between 0 and 1; if `ci_method` is neither of the two
        accepted values; or if `n_boot` is not positive.

    See Also
    --------
    model2_regression : The four-value form of this fit.
    continuous_representativeness : Applies it per target radius.
    classify_continuous : Turns slope, intercept, and R-squared into a Level.

    Notes
    -----
    **Ordinary least squares must not be substituted here.** OLS assumes the
    predictor is measured without error and minimises only the vertical
    residuals, so its slope is ``r`` times this one: with the R-squared of
    0.94 down to 0.71 that Chu et al. (2021) report across target radii, an
    OLS slope would sit roughly 3 % to 16 % below the RMA slope on the same
    data -- always shallower, never steeper. Both variables here are spatial
    averages of the same noisy raster and both carry error, so that
    attenuation is a bias, not a difference of convention. It runs the wrong
    way for this analysis twice over: it deepens the apparent
    footprint-to-target-area bias of Sect. 3.3, and it pushes sites out of the
    ``0.9 <= slope <= 1.1`` HIGH criterion of Sect. 2.4 for no reason but the
    scatter. Table 1 is reproducible only against ``lmodel2``'s RMA.

    R-squared and the p-value are those of the Pearson correlation. They are
    symmetric in `x` and `y` and say nothing about the RMA line's own
    residuals; the RMA slope cannot be zero, so the correlation's null is the
    only meaningful one to test, which is what ``lmodel2`` reports.

    Pairs where either value is non-finite are dropped before fitting. The
    slope is returned as ``nan`` when either variable is constant, or when the
    correlation is exactly zero and its sign cannot be taken -- ``lmodel2``
    likewise refuses an SMA slope it cannot orient. R-squared and the p-value
    are still reported in the second case.

    Examples
    --------
    >>> footprint = [0.21, 0.35, 0.52, 0.66, 0.71, 0.44]
    >>> target = [0.19, 0.33, 0.49, 0.62, 0.70, 0.41]
    >>> fit = rma_regression(footprint, target)
    >>> fit.slope_ci[0] < fit.slope < fit.slope_ci[1]
    True

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Eq. 7.

    Legendre, P. (2014). lmodel2: Model II Regression. R package version 1.7-2.

    McArdle, B. H. (1988). The structural relationship: regression in biology.
    *Canadian Journal of Zoology*, **66**, 2329-2339. Source of the analytical
    interval.
    """
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError(
            f"ci_level must lie strictly between 0 and 1, got {ci_level!r}. "
            f"Pass 0.95 for the 95 % intervals of Table 1."
        )
    if ci_method not in ("analytical", "bootstrap"):
        raise ValueError(
            f"ci_method must be 'analytical' or 'bootstrap', got {ci_method!r}."
        )
    if ci_method == "bootstrap" and int(n_boot) < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot!r}.")

    xs, ys = _finite_pairs(x, y)
    n = int(xs.size)
    if n < 3:
        raise ValueError(
            f"A model II regression needs at least three finite pairs, got "
            f"{n}. Chu et al. (2021) required six matched months before "
            f"fitting a site at all; see continuous_representativeness."
        )

    level = float(ci_level)
    if xs.std() == 0.0 or ys.std() == 0.0:
        # No correlation is defined against a constant, so scipy would warn
        # and hand back nan; say so directly instead.
        r = float("nan")
        p_value = float("nan")
    else:
        correlation = pearsonr(xs, ys)
        r = float(correlation[0])
        p_value = float(correlation[1])

    slope_array, intercept_array = _rma_slope_intercept(xs, ys)
    slope = float(slope_array)
    intercept = float(intercept_array)
    r_squared = float(r**2) if np.isfinite(r) else float("nan")

    if not np.isfinite(slope):
        slope_ci = (float("nan"), float("nan"))
        intercept_ci = (float("nan"), float("nan"))
    elif ci_method == "analytical":
        # McArdle (1988), as reported by lmodel2: the slope limits are the
        # point estimate scaled by sqrt(B + 1) -/+ sqrt(B).
        t_crit = float(_student_t.ppf(0.5 + level / 2.0, n - 2))
        b = t_crit**2 * (1.0 - r_squared) / (n - 2)
        scale = np.array(
            [np.sqrt(b + 1.0) - np.sqrt(b), np.sqrt(b + 1.0) + np.sqrt(b)]
        )
        limits = np.sort(slope * scale)
        slope_ci = (float(limits[0]), float(limits[1]))
        # The intercept is monotone in the slope through the centroid, so its
        # limits are the slope's, reordered when the centroid flips them.
        bounds = np.sort(ys.mean() - limits * xs.mean())
        intercept_ci = (float(bounds[0]), float(bounds[1]))
    else:
        rng = np.random.default_rng(random_state)
        draws = rng.integers(0, n, size=(int(n_boot), n))
        boot_slope, boot_intercept = _rma_slope_intercept(xs[draws], ys[draws])
        tail = 100.0 * (1.0 - level) / 2.0
        if np.count_nonzero(np.isfinite(boot_slope)) < 2:
            # Every resample was degenerate -- a near-constant variable, or
            # too few distinct values to resample away from one.
            slope_ci = (float("nan"), float("nan"))
            intercept_ci = (float("nan"), float("nan"))
        else:
            low, high = np.nanpercentile(boot_slope, [tail, 100.0 - tail])
            slope_ci = (float(low), float(high))
            low, high = np.nanpercentile(boot_intercept, [tail, 100.0 - tail])
            intercept_ci = (float(low), float(high))

    return RMAFit(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        n=n,
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        ci_level=level,
        ci_method=ci_method,
    )


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

    This is :func:`rma_regression` reduced to four values; call that directly
    for the confidence intervals of Table 1, and read its note on why ordinary
    least squares cannot stand in for the fit.
    """
    fit = rma_regression(footprint_values, target_values)
    return fit.intercept, fit.slope, fit.r_squared, fit.p_value


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
    # A non-finite p-value -- an undefined test -- fails this comparison and
    # so falls through to LOW, as does a non-finite percentage below.
    if p_value >= alpha:
        if p_footprint >= 80.0 and p_target >= 80.0:
            return Level.HIGH
        if p_footprint >= 50.0 and p_target >= 50.0:
            return Level.MEDIUM
    return Level.LOW


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

    A degenerate fit -- too few distinct values to orient a slope, say --
    arrives here as non-finite statistics, which fail every comparison below
    and so fall through to LOW.
    """
    if r_squared >= 0.8 and 0.9 <= slope <= 1.1 and -0.1 <= intercept <= 0.1:
        return Level.HIGH
    if r_squared >= 0.6 and p_value < alpha:
        return Level.MEDIUM
    return Level.LOW


def _composition_chi_square(
    footprint: pd.Series,
    target: pd.Series,
) -> tuple[float, float, int]:
    """
    Compare a footprint-weighted and a target-area composition, Sect. 2.4.

    Parameters
    ----------
    footprint : pandas.Series
        Footprint-weighted class fractions from
        :func:`footprint_weighted_composition`.
    target : pandas.Series
        Target-area class fractions from :func:`target_area_composition`,
        whose ``attrs["n_cells"]`` supplies the sample size.

    Returns
    -------
    chi2 : float
        Chi-square statistic. ``inf`` when the footprint sees a class the
        target area holds none of; ``nan`` when the target area holds no
        classified cell to test against.
    p_value : float
        Upper-tail probability. ``0.0`` alongside an infinite statistic,
        ``1.0`` when both compositions consist of the same single class, and
        ``nan`` alongside a ``nan`` statistic.
    dof : int
        Degrees of freedom, one less than the number of classes tested.

    Notes
    -----
    A chi-square test needs counts, but a footprint-weighted composition is a
    continuous share of source weight with no natural sample size -- the cells
    under it are not independent draws, and their count depends on the grid
    rather than on the landscape. Both compositions are therefore scaled to
    **pseudo-counts** by the same sample size, the number of classified cells
    inside the target area, and the footprint's counts are tested against the
    target area's as expectations. That choice makes the test well posed and
    puts the two compositions on one footing, but it fixes the power of the
    test by the grid: a finer grid raises the cell count, hence the counts,
    hence the statistic, for compositions that have not changed. Read the
    p-value as a comparison against the paper's fixed 30 m Landsat grid, not
    as an absolute significance.

    Classes absent from both compositions are dropped rather than entered as
    matching zeros. Keeping them would inflate the degrees of freedom with
    classes the analysis never saw -- a raster carrying all 16 consolidated
    groups of Table S6 would spend degrees of freedom on the dozen a given
    site holds nowhere -- which weakens the test regardless of the landscape.
    Taking the union of the two indices drops them by construction.
    """
    n_cells = int(target.attrs.get("n_cells", 0))
    classes = footprint.index.union(target.index)
    dof = int(classes.size) - 1

    if n_cells <= 0 or classes.size == 0:
        return float("nan"), float("nan"), max(dof, 0)

    observed = (
        footprint.reindex(classes, fill_value=0.0).to_numpy(dtype=float) * n_cells
    )
    expected = target.reindex(classes, fill_value=0.0).to_numpy(dtype=float) * n_cells

    if dof == 0:
        # One class, held by both compositions in full: nothing can differ.
        return 0.0, 1.0, 0
    if np.any(expected <= 0.0):
        # A class the footprint sees and the target area holds none of. Its
        # term is unbounded, so the compositions cannot be reconciled.
        return float("inf"), 0.0, dof

    result = chisquare(observed, expected)
    return float(result.statistic), float(result.pvalue), dof


def categorical_representativeness(
    w: xr.DataArray | np.ndarray,
    landcover: xr.DataArray | np.ndarray,
    x: np.ndarray | xr.DataArray,
    y: np.ndarray | xr.DataArray,
    radii: Sequence[float] = TARGET_RADII,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """
    Evaluate land-cover representativeness against each target area, Sect. 2.4.

    The categorical counterpart of :func:`sensor_location_bias`. The dominant
    land-cover type -- the class holding the largest footprint-weighted share
    -- is identified once, since it does not depend on the target area; its
    share within each disc is then read off, the full compositions are
    compared with a chi-square test, and each radius is classified HIGH,
    MEDIUM, or LOW by :func:`classify_categorical`.

    Parameters
    ----------
    w : xarray.DataArray or numpy.ndarray
        Footprint source weights with dims ``(x, y)``, normally a truncated,
        renormalised climatology from :func:`truncate_to_contour` or a slice
        of :func:`monthly_climatologies`. Weights summing to something other
        than 1 are renormalised, as in :func:`footprint_weighted_composition`.
    landcover : xarray.DataArray or numpy.ndarray
        Categorical raster on the same grid, typically the integer codes of
        the consolidated NLCD / Land Cover of Canada groups of Table S6, NaN
        where it holds no data. Use :func:`sample_raster_on_grid` with
        ``categorical=True`` to bring an external product onto the grid.
    x, y : numpy.ndarray or xarray.DataArray
        Cell-centre offsets from the tower [m] (``model.x``, ``model.y``),
        which is the grid both `w` and `landcover` must be on.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m], evaluated and reported in the order given.
    alpha : float, default 0.05
        Significance level; compositions with ``p_value >= alpha`` count as
        not significantly different.

    Returns
    -------
    pandas.DataFrame
        One row per radius, with columns

        ``radius``
            Target-area radius [m].
        ``dominant_class``
            Class code holding the largest footprint-weighted share. Constant
            down the frame, since the footprint does not depend on the radius.
        ``p_footprint``
            That class's footprint-weighted share, P_footprint, **as a
            fraction in [0, 1]** rather than the percentage of the paper;
            multiply by 100 to match its figures. Constant down the frame.
        ``p_target``
            The same class's share of the disc, P_target, likewise a fraction.
            ``0.0`` when the disc holds none of it.
        ``chi2``
            Chi-square statistic comparing the two full compositions.
        ``p_value``
            p-value of that test.
        ``dof``
            Degrees of freedom, one less than the number of classes tested.
        ``level``
            :class:`Level` member: ``HIGH`` when both shares reach 0.80 and
            the compositions do not differ significantly, ``MEDIUM`` when both
            reach 0.50 and they do not differ significantly, ``LOW``
            otherwise. Being a ``str`` enum it compares equal to ``"high"``
            and serialises as it.

        ``DataFrame.attrs`` holds ``alpha`` and ``footprint_composition``, the
        full footprint-weighted composition the dominant class was taken from.

    Raises
    ------
    ValueError
        If `radii` is empty; if any radius is not positive and finite, or is
        smaller than half the grid spacing so that its disc holds no cell
        centre; if `w`, `landcover`, and the grid `x` and `y` describe are not
        one and the same; if `w` holds a non-finite or negative weight, or no
        weight at all; or if no cell carries both weight and a class code,
        leaving no composition to take a dominant class from.

    See Also
    --------
    footprint_weighted_composition : P_footprint over the whole composition.
    target_area_composition : P_target over the whole composition.
    classify_categorical : The three-level index applied here.
    sensor_location_bias : The continuous-field counterpart, Eq. 6.

    Notes
    -----
    The chi-square test rests on pseudo-counts: both compositions are scaled
    by the number of classified cells inside the target area, because a
    footprint-weighted composition is a share of source weight and carries no
    sample size of its own. Classes absent from both compositions are dropped
    rather than entered as matching zeros, so the degrees of freedom count
    only the classes the analysis actually saw. Both choices are set out in
    :func:`_composition_chi_square`; the p-value is best read as a comparison
    on the paper's 30 m grid rather than as an absolute significance.

    The dominant class comes from the footprint, never from the target area,
    so ``p_target`` can fall to zero while a different class dominates the
    disc -- the mismatch the index is there to report. Ties for the largest
    footprint share go to the lowest class code.

    A raster covering only part of the footprint or the disc lowers the
    retained weight of the underlying compositions rather than making them
    NaN, so a row here can rest on partial coverage; read
    ``attrs["footprint_composition"].attrs["retained_weight"]`` when that
    matters.

    Chu et al. (2021) found 34 of 214 sites where the land-cover product
    disagreed with the site's own IGBP metadata, so a result that contradicts
    site metadata is worth checking against the product before trusting it.

    Examples
    --------
    >>> nlcd = sample_raster_on_grid(                          # doctest: +SKIP
    ...     "nlcd.tif", model.x, model.y, lat, lon, categorical=True
    ... )
    >>> categorical_representativeness(                        # doctest: +SKIP
    ...     truncate_to_contour(model.fclim_2d), nlcd, model.x, model.y
    ... )[["radius", "dominant_class", "p_target", "level"]]
       radius  dominant_class  p_target   level
    0   250.0              41     0.912    high
    1   500.0              41     0.774  medium
    ...

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.

    Goeckede, M., et al. (2008). Quality control of CarboEurope flux data --
    Part 1. *Biogeosciences*, **5**, 433-450. Source of the 50 % and 80 %
    criteria.
    """
    radii_values = [float(radius) for radius in radii]
    if not radii_values:
        raise ValueError(
            "radii holds no target areas, so there is nothing to compare the "
            "footprint against. Pass at least one radius, e.g. TARGET_RADII."
        )

    footprint = footprint_weighted_composition(w, landcover)
    if footprint.empty:
        raise ValueError(
            "No cell carries both source weight and a land-cover code, so the "
            "footprint has no composition and no dominant class. Check that "
            "landcover holds data over the footprint rather than only NaN."
        )

    dominant = footprint.idxmax()
    p_footprint = float(footprint.loc[dominant])

    rows = []
    for radius in radii_values:
        target = target_area_composition(landcover, x, y, radius)
        chi2, p_value, dof = _composition_chi_square(footprint, target)
        p_target = float(target.get(dominant, 0.0))
        rows.append(
            {
                "radius": radius,
                "dominant_class": dominant,
                "p_footprint": p_footprint,
                "p_target": p_target,
                "chi2": chi2,
                "p_value": p_value,
                "dof": dof,
                # classify_categorical works in the percentages of the paper,
                # the columns above in fractions.
                "level": classify_categorical(
                    100.0 * p_footprint, 100.0 * p_target, p_value, alpha
                ),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "radius",
            "dominant_class",
            "p_footprint",
            "p_target",
            "chi2",
            "p_value",
            "dof",
            "level",
        ],
    )
    frame = frame.astype(
        {
            "radius": float,
            "p_footprint": float,
            "p_target": float,
            "chi2": float,
            "p_value": float,
            "dof": int,
        }
    )
    frame.attrs["alpha"] = float(alpha)
    frame.attrs["footprint_composition"] = footprint
    return frame


#: Columns :func:`continuous_representativeness` expects on its input frame.
_PAIR_COLUMNS: tuple[str, ...] = ("time", "radius", "value_footprint", "value_target")


def continuous_representativeness(
    pairs: pd.DataFrame,
    radii: Sequence[float] = TARGET_RADII,
    min_matches: int = MIN_MATCHES,
    alpha: float = DEFAULT_ALPHA,
    ci_level: float = 0.95,
    ci_method: str = "analytical",
    n_boot: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Evaluate continuous-field representativeness per target area, Sect. 2.4.

    The continuous counterpart of :func:`categorical_representativeness`, and
    the site-level step that follows :func:`sensor_location_bias_series`: the
    matched periods of one site are regressed against each target radius by
    Eq. 7 and classified HIGH, MEDIUM, or LOW by :func:`classify_continuous`.

    .. math:: EVI_{target} \\sim \\beta_0 + \\beta_1 EVI_{footprint}

    Parameters
    ----------
    pairs : pandas.DataFrame
        Tidy frame of matched periods, one row per period and radius, with
        columns

        ``time``
            Period label, carried only to identify the matches; the fit does
            not use it and does not sort by it.
        ``radius``
            Target-area radius [m].
        ``value_footprint``
            Footprint-weighted value, EVI_footprint of Eq. 5.
        ``value_target``
            Target-area mean over the disc of that radius, EVI_target.

        Exactly the frame :func:`sensor_location_bias_series` returns; any
        further columns, ``delta`` and ``within_threshold`` among them, are
        ignored.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m] to report, in the order given. Radii of `pairs`
        not listed here are ignored.
    min_matches : int, default 6
        Fewest matched periods a radius may be fitted on. Below it the row is
        reported with ``sufficient = False`` and a missing `level` rather than
        a fitted one, following the paper, which regressed only the 166 of 214
        sites holding at least six matches. Must be at least 3.
    alpha : float, default 0.05
        Significance level for the MEDIUM criterion.
    ci_level : float, default 0.95
        Confidence level of the reported interval limits.
    ci_method : {'analytical', 'bootstrap'}, default 'analytical'
        Interval estimator, passed to :func:`rma_regression`.
    n_boot : int, default 2000
        Number of resamples, used only when ``ci_method='bootstrap'``.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or generator for the bootstrap.

    Returns
    -------
    pandas.DataFrame
        One row per requested radius, in the order given, with columns

        ``radius``
            Target-area radius [m].
        ``n``
            Matched periods entering the row: rows of `pairs` at that radius
            holding a finite value on both sides.
        ``intercept``, ``slope``
            beta_0 and beta_1 of Eq. 7. A slope below 1 means the footprint
            saw systematically higher values than the target area, which held
            at every radius in the paper.
        ``r_squared``, ``p_value``
            Squared Pearson correlation and its two-sided significance.
        ``rmse``, ``mae``
            Root mean square and mean absolute *difference* between the
            footprint-weighted and target-area values, as in Table 1. These
            describe the two series against the 1:1 line, not the residuals
            about the fitted line, and so are reported whenever the radius
            holds a match at all.
        ``intercept_lower``, ``intercept_upper``, ``slope_lower``, ``slope_upper``
            Confidence limits at `ci_level`, the square-bracketed columns of
            Table 1.
        ``sufficient``
            Whether ``n >= min_matches``, i.e. whether the row was fitted.
        ``level``
            :class:`Level` member, or ``pd.NA`` where `sufficient` is False.
            ``HIGH`` when ``r_squared >= 0.8`` with slope in [0.9, 1.1] and
            intercept in [-0.1, 0.1]; ``MEDIUM`` when ``r_squared >= 0.6`` and
            ``p_value < alpha``; ``LOW`` otherwise.

        ``DataFrame.attrs`` records ``min_matches``, ``alpha``, ``ci_level``,
        and ``ci_method``.

    Raises
    ------
    KeyError
        If `pairs` is missing any of the four required columns.
    ValueError
        If `pairs` is not a DataFrame; if `radii` is empty or holds a radius
        that is not positive and finite; or if `min_matches` is below 3, which
        is the fewest pairs a regression can be fitted on at all.

    See Also
    --------
    sensor_location_bias_series : Produces the frame this consumes.
    rma_regression : The model II fit applied per radius.
    classify_continuous : The three-level index applied here.
    categorical_representativeness : The land-cover counterpart.

    Notes
    -----
    A radius listed in `radii` but absent from `pairs` comes back with
    ``n = 0`` and ``sufficient = False`` rather than raising, so a frame built
    over a shorter set of radii still reports against the full series. Radii
    are matched to within floating-point tolerance, so an integer 250 finds a
    250.0 in the frame.

    The insufficient rows are the point of `min_matches`: a site-radius with
    three or four matches can be fitted, and will be classified LOW or HIGH by
    the same criteria, but on evidence too thin to mean what the index says.
    Chu et al. (2021) set the bar at six matched months, with a median of 13
    per site. Filter on ``sufficient`` rather than on ``level.notna()`` if the
    distinction matters downstream.

    Rows are fitted independently per radius. That is deliberate: the
    footprint-weighted value is the same across radii while the target-area
    mean is not, so the systematic bias grows with the disc -- from a slope of
    0.96 at 250 m to 0.80 at 3000 m in Table 1 -- and pooling radii would
    average that trend away.

    Examples
    --------
    >>> bias = sensor_location_bias_series(scenes, model.x, model.y)  # doctest: +SKIP
    >>> site = continuous_representativeness(bias)                   # doctest: +SKIP
    >>> site[["radius", "n", "slope", "r_squared", "level"]]         # doctest: +SKIP
       radius   n  slope  r_squared   level
    0   250.0  13  0.962      0.941    high
    1   500.0  13  0.913      0.883  medium
    ...

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4,
    Table 1.
    """
    if not isinstance(pairs, pd.DataFrame):
        raise ValueError(
            f"pairs must be a tidy DataFrame of matched periods with columns "
            f"{list(_PAIR_COLUMNS)}, got {type(pairs).__name__}. "
            f"sensor_location_bias_series returns one."
        )
    missing = [column for column in _PAIR_COLUMNS if column not in pairs.columns]
    if missing:
        raise KeyError(
            f"pairs is missing the column(s) {missing}; a row of matched "
            f"periods needs {list(_PAIR_COLUMNS)}."
        )
    if int(min_matches) < 3:
        raise ValueError(
            f"min_matches must be at least 3, the fewest pairs a regression "
            f"can be fitted on, got {min_matches!r}. Chu et al. (2021) used 6."
        )

    radii_values = [float(radius) for radius in radii]
    if not radii_values:
        raise ValueError(
            "radii holds no target areas, so there is nothing to compare the "
            "footprint against. Pass at least one radius, e.g. TARGET_RADII."
        )
    for radius in radii_values:
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"Every target radius must be positive and finite, got "
                f"{radius!r}."
            )

    threshold = int(min_matches)
    frame_radii = pairs["radius"].to_numpy(dtype=float)
    footprint_all = pairs["value_footprint"].to_numpy(dtype=float)
    target_all = pairs["value_target"].to_numpy(dtype=float)

    rows = []
    for radius in radii_values:
        here = np.isclose(frame_radii, radius, rtol=1e-9, atol=0.0)
        footprint, target = _finite_pairs(footprint_all[here], target_all[here])
        n = int(footprint.size)

        row: dict[str, Any] = {
            "radius": radius,
            "n": n,
            "intercept": float("nan"),
            "slope": float("nan"),
            "r_squared": float("nan"),
            "p_value": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "intercept_lower": float("nan"),
            "intercept_upper": float("nan"),
            "slope_lower": float("nan"),
            "slope_upper": float("nan"),
            "sufficient": n >= threshold,
            "level": pd.NA,
        }

        if n:
            # Against the 1:1 line, as in Table 1, not about the fitted line.
            difference = target - footprint
            row["rmse"] = float(np.sqrt(np.mean(difference**2)))
            row["mae"] = float(np.mean(np.abs(difference)))

        if row["sufficient"]:
            fit = rma_regression(
                footprint,
                target,
                ci_level=ci_level,
                ci_method=ci_method,
                n_boot=n_boot,
                random_state=random_state,
            )
            row.update(
                intercept=fit.intercept,
                slope=fit.slope,
                r_squared=fit.r_squared,
                p_value=fit.p_value,
                intercept_lower=fit.intercept_ci[0],
                intercept_upper=fit.intercept_ci[1],
                slope_lower=fit.slope_ci[0],
                slope_upper=fit.slope_ci[1],
                level=classify_continuous(
                    fit.r_squared, fit.slope, fit.intercept, fit.p_value, alpha
                ),
            )
        rows.append(row)

    results = pd.DataFrame(
        rows,
        columns=[
            "radius",
            "n",
            "intercept",
            "slope",
            "r_squared",
            "p_value",
            "rmse",
            "mae",
            "intercept_lower",
            "intercept_upper",
            "slope_lower",
            "slope_upper",
            "sufficient",
            "level",
        ],
    )
    results = results.astype(
        {
            "radius": float,
            "n": int,
            "intercept": float,
            "slope": float,
            "r_squared": float,
            "p_value": float,
            "rmse": float,
            "mae": float,
            "intercept_lower": float,
            "intercept_upper": float,
            "slope_lower": float,
            "slope_upper": float,
            "sufficient": bool,
        }
    )
    results.attrs["min_matches"] = threshold
    results.attrs["alpha"] = float(alpha)
    results.attrs["ci_level"] = float(ci_level)
    results.attrs["ci_method"] = ci_method
    return results


# ------------------------------
# Site-level evaluation
# ------------------------------


def _ensure_truncated(
    fclim: xr.DataArray,
    dx: float | None,
    dy: float | None,
    fraction: float,
) -> xr.DataArray:
    """
    Return a climatology as unit-sum weights, truncating it only if needed.

    Parameters
    ----------
    fclim : xarray.DataArray
        A raw climatology with dims ``(x, y)`` [m⁻²], or one already cut by
        :func:`truncate_to_contour`.
    dx, dy : float or None
        Grid spacing [m]. Inferred from coordinates when omitted.
    fraction : float
        Source-weight fraction to truncate a raw climatology at.

    Returns
    -------
    xarray.DataArray
        Weights summing to 1 over the retained cells.

    Raises
    ------
    ValueError
        If the array carries no positive weight to rescale.

    Notes
    -----
    Truncated input is recognised by the ``contour_fraction`` attribute, as in
    :func:`climatology_metrics`, and is rescaled rather than cut a second
    time: a second pass would take 80 % of an already-80 % source area and
    quietly shrink it. A slice of :func:`monthly_climatologies` arrives
    already truncated *and* already summing to 1, and comes back untouched.
    """
    if "contour_fraction" not in fclim.attrs:
        return truncate_to_contour(fclim, dx, dy, fraction=fraction, renormalize=True)

    finite = fclim.where(np.isfinite(fclim), 0.0)
    total = float(finite.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "The truncated climatology carries no positive weight, so it "
            "cannot be renormalised."
        )
    if abs(total - 1.0) <= _NORMALIZATION_TOLERANCE:
        return fclim
    return (finite / total).assign_attrs(fclim.attrs)


def _evaluation_grid(fclim: xr.DataArray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the tower-centred cell-centre offsets off a climatology.

    Parameters
    ----------
    fclim : xarray.DataArray
        Climatology carrying ``x`` and ``y`` coordinates [m].
    name : str
        How to name the array in the error message.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``, one-dimensional and in metres from the tower.

    Raises
    ------
    ValueError
        If either coordinate is missing, since the target-area discs are laid
        out on them.
    """
    for axis in ("x", "y"):
        if axis not in fclim.coords:
            raise ValueError(
                f"{name} carries no '{axis}' coordinate, so the target areas "
                f"cannot be placed on it. Pass a climatology with the "
                f"tower-centred ({{'x'}}, {{'y'}}) coordinates the models produce."
            )
    return (
        np.asarray(fclim.coords["x"].values, dtype=float),
        np.asarray(fclim.coords["y"].values, dtype=float),
    )


def _as_percentages(composition: pd.Series) -> dict[Any, float]:
    """
    Convert a composition of weight fractions into the paper's percentages.

    Parameters
    ----------
    composition : pandas.Series
        Class code -> share in [0, 1], from
        :func:`footprint_weighted_composition` or
        :func:`target_area_composition`.

    Returns
    -------
    dict
        The same mapping in percent, as :class:`CategoricalResult` reports it.
    """
    return {code: 100.0 * float(share) for code, share in composition.items()}


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
        One entry per radius, in the order given. The shares and both
        compositions are percentages, as the paper reports them, not the
        fractions :func:`categorical_representativeness` returns.

    Raises
    ------
    ValueError
        If `fclim` and `classes` are not on the same grid; if `fclim` carries
        no ``x`` or ``y`` coordinate to lay the discs out on; if `radii` is
        empty or holds an unusable radius; or if no cell carries both source
        weight and a class code.

    See Also
    --------
    categorical_representativeness : The per-radius comparison this wraps.
    assess_representativeness : The whole-record driver, which reports the
        same comparison as tidy rows for every month and period.
    evaluate_vegetation_index : The continuous-field counterpart.
    representativeness_summary : Flattens these results into a table.

    Notes
    -----
    The chi-square test compares footprint-weighted counts against target-area
    counts; classes absent from both are dropped. The paper found 34 of 214
    sites where the land-cover product disagreed with the site's own IGBP
    metadata, so a disagreement between this result and site metadata is worth
    checking against the product before trusting it.

    A raw climatology is truncated here at `fraction` and rescaled to unit
    sum; one already cut by :func:`truncate_to_contour` is rescaled but not
    cut again, so a slice of :func:`monthly_climatologies` can be passed
    straight in.

    Examples
    --------
    >>> nlcd = sample_raster_on_grid(                        # doctest: +SKIP
    ...     "nlcd.tif", model.x, model.y, lat, lon, categorical=True
    ... )
    >>> results = evaluate_landcover(model.fclim_2d, nlcd)   # doctest: +SKIP
    >>> [(entry.radius, entry.level) for entry in results]   # doctest: +SKIP
    [(250.0, <Level.HIGH: 'high'>), (500.0, <Level.MEDIUM: 'medium'>), ...]

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """
    x, y = _evaluation_grid(fclim, "fclim")
    weights = _ensure_truncated(fclim, dx, dy, fraction)

    frame = categorical_representativeness(
        weights, classes, x, y, radii=radii, alpha=alpha
    )
    footprint = _as_percentages(frame.attrs["footprint_composition"])

    results: list[CategoricalResult] = []
    for record in frame.itertuples(index=False):
        # categorical_representativeness reports the dominant class's share
        # but not the whole disc, which CategoricalResult carries.
        target = target_area_composition(classes, x, y, record.radius)
        results.append(
            CategoricalResult(
                radius=float(record.radius),
                dominant_class=record.dominant_class,
                p_footprint=100.0 * float(record.p_footprint),
                p_target=100.0 * float(record.p_target),
                chi2=float(record.chi2),
                p_value=float(record.p_value),
                dof=int(record.dof),
                level=Level(record.level),
                footprint_composition=dict(footprint),
                target_composition=_as_percentages(target),
            )
        )
    return results


def evaluate_vegetation_index(
    climatologies: Sequence[xr.DataArray],
    fields: Sequence[xr.DataArray],
    radii: Sequence[float] = TARGET_RADII,
    dx: float | None = None,
    dy: float | None = None,
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    bias_threshold: float = BIAS_THRESHOLD,
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
        One entry per radius, in the order given. ``bias`` holds the biases of
        the periods that entered that radius's regression, and ``level`` is
        None where fewer than three of them survived -- a raster covering only
        part of the larger discs can leave a radius unfittable while the
        smaller ones are fine.

    Raises
    ------
    ValueError
        If the two sequences differ in length; if fewer than three pairs are
        supplied; if the first climatology carries no ``x`` or ``y``
        coordinate; or if a climatology and its field are not on one grid.

    See Also
    --------
    sensor_location_bias : Eq. 6, computed once per pair here.
    continuous_representativeness : The per-radius regression this wraps.
    assess_representativeness : The whole-record driver, which matches scenes
        to months itself and reports the per-period biases alongside the fit.
    evaluate_landcover : The categorical counterpart.
    representativeness_summary : Flattens these results into a table.

    Notes
    -----
    The paper required at least six matched pairs for a site-level regression,
    with a median of 13 per site. Fewer pairs will fit but the resulting level
    should not be trusted.

    Pairing is the caller's job: the two sequences are matched element by
    element, in the order given. The paper pairs each Landsat scene with the
    climatology of the month it was retrieved in.

    ``within_threshold`` averages the flags of the periods whose bias is
    defined, so a scene that did not reach the disc lowers the count of
    periods rather than counting as a failure -- the convention behind the
    percentages of Sect. 3.3 and Fig. 7.

    Examples
    --------
    >>> results = evaluate_vegetation_index(clims, scenes)   # doctest: +SKIP
    >>> [(entry.radius, round(entry.slope, 2)) for entry in results]
    [(250.0, 0.96), (500.0, 0.91), ...]                      # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4,
    Table 1.
    """
    clims = list(climatologies)
    rasters = list(fields)
    if len(clims) != len(rasters):
        raise ValueError(
            f"climatologies and fields must be matched element for element, "
            f"got {len(clims)} climatologies and {len(rasters)} fields."
        )
    if len(clims) < 3:
        raise ValueError(
            f"A regression needs at least three matched pairs, got "
            f"{len(clims)}. Chu et al. (2021) used six."
        )

    x, y = _evaluation_grid(clims[0], "The first climatology")

    frames: list[pd.DataFrame] = []
    for index, (fclim, field) in enumerate(zip(clims, rasters)):
        weights = _ensure_truncated(fclim, dx, dy, fraction)
        try:
            frame = sensor_location_bias(weights, field, x, y, radii=radii)
        except ValueError as exc:
            raise ValueError(f"At pair {index}: {exc}") from exc
        frame.insert(0, "time", index)
        frames.append(frame)

    pairs = pd.concat(frames, ignore_index=True)
    # min_matches=3 so every radius is fitted; the docstring, not the fit,
    # carries the paper's warning about resting on fewer than six.
    fits = continuous_representativeness(pairs, radii=radii, min_matches=3, alpha=alpha)

    pair_radii = pairs["radius"].to_numpy(dtype=float)
    footprint_values = pairs["value_footprint"].to_numpy(dtype=float)
    target_values = pairs["value_target"].to_numpy(dtype=float)
    deltas = pairs["delta"].to_numpy(dtype=float)

    results: list[ContinuousResult] = []
    for record in fits.itertuples(index=False):
        here = (
            np.isclose(pair_radii, record.radius, rtol=1e-9, atol=0.0)
            & np.isfinite(footprint_values)
            & np.isfinite(target_values)
        )
        bias = deltas[here]
        flags = _within_threshold(bias, bias_threshold)
        within = float(pd.Series(flags).mean()) if bias.size else float("nan")

        results.append(
            ContinuousResult(
                radius=float(record.radius),
                intercept=float(record.intercept),
                slope=float(record.slope),
                r_squared=float(record.r_squared),
                p_value=float(record.p_value),
                rmse=float(record.rmse),
                mae=float(record.mae),
                n=int(record.n),
                level=None if pd.isna(record.level) else Level(record.level),
                bias=bias,
                within_threshold=within,
            )
        )
    return results


#: Columns of the frame :func:`representativeness_summary` returns, in order.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "site_id",
    "radius",
    "dominant_class",
    "p_footprint",
    "p_target",
    "chi2",
    "chi2_p_value",
    "dof",
    "landcover_level",
    "n",
    "slope",
    "intercept",
    "r_squared",
    "regression_p_value",
    "rmse",
    "mae",
    "mean_bias",
    "median_bias",
    "bias_within_threshold",
    "continuous_level",
    "contour_fraction",
    "contour_level",
    "fetch",
    "area",
    "symmetry",
    "enclosed_fraction",
    "n_cells",
    "seasonal_overlap",
    "daynight_overlap",
)

#: Summary columns held as pandas' nullable integer.
_SUMMARY_INT_COLUMNS: tuple[str, ...] = ("dof", "n", "n_cells")


def representativeness_summary(
    categorical: Sequence[CategoricalResult] | None = None,
    continuous: Sequence[ContinuousResult] | None = None,
    metrics: ClimatologyMetrics | None = None,
    site_id: str | None = None,
) -> pd.DataFrame:
    """
    Flatten representativeness results into a tidy table.

    Joins the two halves of the analysis on their target-area radius, so that
    a site's land-cover and vegetation-index verdicts can be read on one line
    per radius, and written or concatenated across sites.

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
        The column is always present, and missing when no identifier is given.

    Returns
    -------
    pandas.DataFrame
        One row per target radius over a fresh :class:`~pandas.RangeIndex`,
        carrying :data:`SUMMARY_COLUMNS` in that order:

        ``site_id``, ``radius``
            Row identity. Radii come in the order `categorical` presents them,
            followed by any radius only `continuous` holds.
        ``dominant_class``, ``p_footprint``, ``p_target``, ``chi2``, ``chi2_p_value``, ``dof``, ``landcover_level``
            The land-cover half, with the two shares as percentages.
        ``n``, ``slope``, ``intercept``, ``r_squared``, ``regression_p_value``, ``rmse``, ``mae``, ``mean_bias``, ``median_bias``, ``bias_within_threshold``, ``continuous_level``
            The continuous half. The three bias columns summarise the
            per-period Deltas of Eq. 6 as fractions; the rest is the model II
            regression of Eq. 7.
        ``contour_fraction`` through ``daynight_overlap``
            The climatology metrics of Sect. 2.2, repeated down the frame
            because they describe the footprint rather than any one disc.

        Missing halves are filled with NaN, so the frame can be concatenated
        across sites and written with
        :func:`~fluxfootprints.export_contour_stats_csv` conventions.

    Raises
    ------
    ValueError
        If both `categorical` and `continuous` are omitted or empty, leaving
        no radius to build rows from, or if either holds two results for the
        same radius.

    See Also
    --------
    evaluate_landcover : Produces the land-cover half.
    evaluate_vegetation_index : Produces the continuous half.
    assess_representativeness : The whole-record driver, whose tidy frame
        carries the same statistics per month and period rather than pooled.
    export_representativeness_gpkg : Draws this table as nested discs.

    Notes
    -----
    The two ``*_level`` columns hold a :class:`Level`'s *value* -- ``"high"``,
    ``"medium"``, or ``"low"`` -- rather than the member, because ``str()`` of
    a member renders as ``"Level.HIGH"`` and would reach a CSV that way. They
    still compare equal to the members.

    A radius present in one half and not the other still gets a row, with the
    other half's columns missing. That is the case worth seeing rather than
    hiding: the land cover and the vegetation index need not have been
    evaluated over the same series of discs.

    Examples
    --------
    >>> table = representativeness_summary(                  # doctest: +SKIP
    ...     evaluate_landcover(model.fclim_2d, nlcd),
    ...     evaluate_vegetation_index(clims, scenes),
    ...     climatology_metrics(model.fclim_2d),
    ...     site_id="US-MOz",
    ... )
    >>> table[["radius", "landcover_level", "continuous_level"]]  # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """
    landcover = list(categorical or [])
    vegetation = list(continuous or [])
    if not landcover and not vegetation:
        raise ValueError(
            "Both result sequences are empty, so there is no radius to build "
            "a row from. Pass the output of evaluate_landcover, "
            "evaluate_vegetation_index, or both."
        )

    by_radius: dict[float, CategoricalResult] = {}
    for entry in landcover:
        radius = float(entry.radius)
        if radius in by_radius:
            raise ValueError(
                f"categorical holds two results for the {radius:g} m target "
                f"area; a summary carries one row per radius."
            )
        by_radius[radius] = entry

    fits: dict[float, ContinuousResult] = {}
    for entry in vegetation:
        radius = float(entry.radius)
        if radius in fits:
            raise ValueError(
                f"continuous holds two results for the {radius:g} m target "
                f"area; a summary carries one row per radius."
            )
        fits[radius] = entry

    radii = list(by_radius)
    radii.extend(radius for radius in fits if radius not in by_radius)

    rows: list[dict[str, Any]] = []
    for radius in radii:
        row: dict[str, Any] = {column: float("nan") for column in SUMMARY_COLUMNS}
        row.update(
            site_id=pd.NA if site_id is None else str(site_id),
            radius=radius,
            dominant_class=pd.NA,
            landcover_level=pd.NA,
            continuous_level=pd.NA,
        )
        row.update({column: pd.NA for column in _SUMMARY_INT_COLUMNS})

        entry = by_radius.get(radius)
        if entry is not None:
            row.update(
                dominant_class=entry.dominant_class,
                p_footprint=float(entry.p_footprint),
                p_target=float(entry.p_target),
                chi2=float(entry.chi2),
                chi2_p_value=float(entry.p_value),
                dof=int(entry.dof),
                landcover_level=_level_label(entry.level),
            )

        fit = fits.get(radius)
        if fit is not None:
            bias = np.asarray(fit.bias, dtype=float)
            finite = bias[np.isfinite(bias)]
            row.update(
                n=int(fit.n),
                slope=float(fit.slope),
                intercept=float(fit.intercept),
                r_squared=float(fit.r_squared),
                regression_p_value=float(fit.p_value),
                rmse=float(fit.rmse),
                mae=float(fit.mae),
                mean_bias=float(finite.mean()) if finite.size else float("nan"),
                median_bias=float(np.median(finite)) if finite.size else float("nan"),
                bias_within_threshold=float(fit.within_threshold),
                continuous_level=_level_label(fit.level),
            )

        if metrics is not None:
            row.update(
                contour_fraction=float(metrics.fraction),
                contour_level=float(metrics.contour_level),
                fetch=float(metrics.fetch),
                area=float(metrics.area),
                symmetry=float(metrics.symmetry),
                enclosed_fraction=float(metrics.enclosed_fraction),
                n_cells=int(metrics.n_cells),
                seasonal_overlap=(
                    float("nan")
                    if metrics.seasonal_overlap is None
                    else float(metrics.seasonal_overlap)
                ),
                daynight_overlap=(
                    float("nan")
                    if metrics.daynight_overlap is None
                    else float(metrics.daynight_overlap)
                ),
            )
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    for column in _SUMMARY_INT_COLUMNS:
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


# ------------------------------
# Optional-dependency helpers
# ------------------------------


def _spatial_dims(da: xr.DataArray, name: str) -> tuple[str, str]:
    """
    Resolve the y and x dimension names rioxarray recognises on an array.

    Parameters
    ----------
    da : xarray.DataArray
        Array expected to carry georeferenced spatial dims.
    name : str
        How to refer to `da` in the error message, phrased to sit mid-sentence,
        e.g. ``"the footprint grid"``.

    Returns
    -------
    tuple of str
        ``(y_dim, x_dim)``.

    Raises
    ------
    ValueError
        If rioxarray cannot identify the spatial dims, with the array's own
        dims in the message.
    """
    rioxarray = _require("rioxarray")
    try:
        return str(da.rio.y_dim), str(da.rio.x_dim)
    except rioxarray.exceptions.MissingSpatialDimensionError as exc:
        raise ValueError(
            f"Cannot find the spatial dimensions of {name}: its dims are "
            f"{tuple(da.dims)}. Name them 'x' and 'y' (or 'longitude' and "
            f"'latitude'), or call .rio.set_spatial_dims() first."
        ) from exc


def _raster_crs(da: xr.DataArray, name: str, hint: str) -> Any:
    """
    Read an array's CRS, or raise saying how to attach one.

    Parameters
    ----------
    da : xarray.DataArray
        Array to read ``.rio.crs`` from.
    name : str
        How to refer to `da` in the error message, phrased to sit mid-sentence,
        e.g. ``"the footprint grid"``.
    hint : str
        Sentence appended to the error, pointing at where the CRS should have
        come from.

    Returns
    -------
    rasterio.crs.CRS
        The array's CRS.

    Raises
    ------
    ValueError
        If the array carries no CRS, so it cannot be reprojected.
    """
    _spatial_dims(da, name)
    crs = da.rio.crs
    if crs is None:
        raise ValueError(f"Cannot reproject {name}: it carries no CRS. {hint}")
    return crs


def _check_metric_crs(crs: Any, name: str) -> None:
    """
    Require a CRS whose axes are metres, as the footprint grid assumes.

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS to check.
    name : str
        How to refer to the array carrying `crs` in the error messages, phrased
        to sit mid-sentence, e.g. ``"the footprint grid"``.

    Raises
    ------
    ValueError
        If the CRS is geographic, or is otherwise not a projected CRS.

    Notes
    -----
    Every length in this module -- fetch, area, target-area radii, cell area --
    is metres. On a geographic grid a cell is degrees wide, so those quantities
    would come out in degrees and vary with latitude, which is a wrong answer
    rather than a failure: hence the up-front check.
    """
    if crs.is_geographic:
        raise ValueError(
            f"Cannot align onto {name}: it is in the geographic CRS "
            f"{crs.to_string()}, whose units are degrees, not metres. The "
            f"representativeness metrics are defined on a metric grid; "
            f"reproject it to a projected CRS such as its local UTM zone "
            f"(see footprint_grid_geometry)."
        )
    if not crs.is_projected:
        raise ValueError(
            f"Cannot align onto {name}: it is in {crs.to_string()}, which is "
            f"not a projected CRS. The representativeness metrics need a "
            f"projected, metric grid (see footprint_grid_geometry)."
        )


def _select_band(source: xr.DataArray, band: int, origin: str) -> xr.DataArray:
    """
    Reduce a raster to the requested band, if it carries several.

    Parameters
    ----------
    source : xarray.DataArray
        Raster as opened, possibly with a leading ``band`` dim.
    band : int
        One-based band index, as rasterio numbers them.
    origin : str
        How to refer to `source` in the error message.

    Returns
    -------
    xarray.DataArray
        A single-band array, with the ``band`` dim dropped.

    Raises
    ------
    ValueError
        If `band` is not a positive integer within the raster's band count.
    """
    if "band" not in source.dims:
        return source
    count = int(source.sizes["band"])
    if not 1 <= band <= count:
        raise ValueError(
            f"band {band} is out of range for {origin}, which has "
            f"{count} band(s); bands are numbered from 1."
        )
    return source.isel(band=band - 1, drop=True)


def _source_nodata(source: xr.DataArray) -> float | None:
    """
    Read a raster's declared nodata value, encoded or in memory.

    Parameters
    ----------
    source : xarray.DataArray
        Raster to inspect.

    Returns
    -------
    float or None
        The nodata value, or None if the raster declares none or already
        represents it as NaN, which needs no substitution.
    """
    for value in (source.rio.nodata, source.rio.encoded_nodata):
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            return value
    return None


def _on_footprint_grid(aligned: xr.DataArray, footprint: xr.DataArray) -> xr.DataArray:
    """
    Put a warped raster on the footprint's own dims and coords.

    Parameters
    ----------
    aligned : xarray.DataArray
        Output of ``reproject_match``, on the footprint's grid but carrying the
        source's dim names and coordinates recomputed from the transform.
    footprint : xarray.DataArray
        Grid that was matched.

    Returns
    -------
    xarray.DataArray
        `aligned` renamed, transposed, and re-coordinated to the footprint, so
        that the two compare equal cell for cell rather than merely closely.

    Notes
    -----
    ``reproject_match`` rebuilds the destination coordinates from the affine
    transform, which can leave them a float ULP away from the footprint's own.
    xarray aligns on coordinate values, so that difference is enough to turn a
    later multiplication into an empty inner join; copying the footprint's
    coordinates across rules it out.
    """
    src_y, src_x = _spatial_dims(aligned, "the reprojected raster")
    dst_y, dst_x = _spatial_dims(footprint, "the footprint grid")

    renames = {src: dst for src, dst in ((src_y, dst_y), (src_x, dst_x)) if src != dst}
    if renames:
        aligned = aligned.rename(renames)

    order = [dim for dim in footprint.dims if dim in (dst_y, dst_x)]
    aligned = aligned.transpose(*order)
    return aligned.assign_coords(
        {dim: footprint.coords[dim] for dim in order if dim in footprint.coords}
    )


def _align_raster(
    raster: xr.DataArray | str | Path,
    footprint: xr.DataArray,
    categorical: bool = False,
    nodata: float | None = None,
    band: int = 1,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Reproject an external raster onto the footprint grid, cell for cell.

    Wraps :meth:`rioxarray.raster_array.RasterArray.reproject_match` so that a
    land-cover or vegetation-index product can be compared against a
    climatology without an intermediate resampling step of the caller's own.
    Continuous fields are resampled bilinearly and categorical ones with
    nearest neighbour, which is the only resampling that leaves class codes
    intact.

    Parameters
    ----------
    raster : xarray.DataArray, str, or pathlib.Path
        Source raster, either already opened with a ``.rio`` accessor or a path
        to anything rasterio can read. Multi-band sources are reduced to one
        band by `band`.
    footprint : xarray.DataArray
        Georeferenced footprint grid to align onto; only its grid is used, not
        its values. It must carry a projected CRS, as written by
        ``.rio.write_crs()`` from
        :func:`~fluxfootprints.footprint_grid_geometry`.
    categorical : bool, default False
        If True, resample with ``Resampling.nearest`` to preserve class codes;
        if False, resample with ``Resampling.bilinear`` for a continuous field.
    nodata : float, optional
        A further value in the source to treat as missing, for rasters that
        use a fill value they do not declare. The source's own declared nodata
        is honoured whether or not this is given.
    band : int, default 1
        One-based band index to align, used only when the source carries a
        ``band`` dimension.

    Returns
    -------
    aligned : xarray.DataArray
        Source values on the footprint's grid, as float64 with the footprint's
        dims and coords, and NaN wherever no valid source data reached the
        cell. Class codes survive the float conversion exactly.
    valid : xarray.DataArray
        Boolean array on the same grid, True where `aligned` holds real data
        and False where the source was nodata or did not cover the cell.

    Raises
    ------
    ImportError
        If ``rioxarray`` is not installed.
    TypeError
        If `raster` is a Dataset rather than a single-variable DataArray, or
        `footprint` is not a DataArray.
    ValueError
        If either array carries no CRS, if their spatial dims cannot be
        identified, if the footprint grid is geographic (degrees) rather than
        projected (metres), or if `band` is out of range.

    Notes
    -----
    Nodata is resolved once in the source's own grid and turned into NaN before
    the warp, so it is the warper that decides how missing data spreads: cells
    the source does not reach come back as NaN, and a bilinear cell touching a
    mix of valid and missing source pixels is interpolated from the valid ones
    alone. Resolving it up front rather than leaning on the source's own
    declaration is also what makes a path and an already-opened DataArray
    behave alike, since :func:`rioxarray.open_rasterio` has usually applied
    the declared nodata already.

    `valid` is exactly ``aligned.notnull()``, returned alongside so that callers
    weighting by footprint mass can renormalise over the covered cells instead
    of silently treating a gap as a zero.

    See Also
    --------
    fluxfootprints.footprint_grid_geometry : Georeferences a footprint grid.
    fluxfootprints.openet_mask_on_grid : The same warp for data-availability masks.

    Examples
    --------
    >>> aligned, valid = _align_raster("evi.tif", grid)  # doctest: +SKIP
    >>> float(aligned.where(valid).mean())               # doctest: +SKIP
    0.42
    """
    rioxarray = _require("rioxarray")
    from rasterio.enums import Resampling

    if isinstance(footprint, xr.Dataset):
        raise TypeError(
            "footprint must be a DataArray carrying the target grid, not a "
            "Dataset. Select the variable holding the climatology first."
        )
    if not isinstance(footprint, xr.DataArray):
        raise TypeError(
            f"footprint must be an xarray.DataArray, got {type(footprint).__name__}."
        )

    if isinstance(raster, (str, Path)):
        # Read eagerly and close: reproject_match would pull the whole source
        # into memory anyway, and a caller aligning a directory of tiles should
        # not be holding a descriptor open for every one of them.
        with rioxarray.open_rasterio(raster, masked=True) as opened:
            source = opened.load()
        origin = f"raster {Path(raster).name}"
    else:
        source = raster
        origin = "raster"
    if isinstance(source, xr.Dataset):
        raise TypeError(
            f"{origin} holds {len(source.data_vars)} variables; pass a single "
            f"DataArray, e.g. ds['evi'], so there is one field to align."
        )

    source = _select_band(source, band, origin)

    dst_crs = _raster_crs(
        footprint,
        "the footprint grid",
        "Georeference it with footprint_grid_geometry() and write the CRS "
        "onto it with .rio.write_crs().",
    )
    _check_metric_crs(dst_crs, "the footprint grid")
    _raster_crs(
        source,
        f"the source {origin}",
        "Set one with .rio.write_crs(), or use a file that carries its own.",
    )

    values = source.astype("float64")
    for fill in (_source_nodata(source), nodata):
        if fill is not None:
            values = values.where(values != fill)
    values.rio.write_nodata(np.nan, inplace=True)

    resampling = Resampling.nearest if categorical else Resampling.bilinear
    aligned = values.rio.reproject_match(
        footprint, resampling=resampling, nodata=np.nan
    )

    aligned = _on_footprint_grid(aligned, footprint)
    aligned.name = getattr(source, "name", None) or "aligned"
    valid = aligned.notnull().rename("valid")
    return aligned, valid


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
    _align_raster : The reprojection this wraps.

    Notes
    -----
    The warp itself happens on the georeferenced grid, whose coordinates are
    projected metres; the result is handed back on the tower-centred ``(x, y)``
    grid the rest of this module works in, so that it lines up cell for cell
    with a climatology and with :func:`target_area_mask`.

    Examples
    --------
    >>> evi = sample_raster_on_grid(              # doctest: +SKIP
    ...     "evi.tif", model.x, model.y, 40.0, -111.9
    ... )
    >>> footprint_weighted_value(weights, evi).value   # doctest: +SKIP
    0.42
    """
    _require("rioxarray")  # registers the .rio accessor used below

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    geometry = footprint_grid_geometry(
        x_values, y_values, station_lat, station_lon, crs=crs
    )
    grid = xr.DataArray(
        np.zeros((y_values.size, x_values.size)),
        dims=("y", "x"),
        coords={
            "y": geometry.y_origin + y_values,
            "x": geometry.x_origin + x_values,
        },
        name="footprint",
    ).rio.write_crs(geometry.crs)

    aligned, _ = _align_raster(raster_path, grid, categorical=categorical, band=band)
    return xr.DataArray(
        aligned.transpose("x", "y").values,
        coords={"x": x_values, "y": y_values},
        dims=("x", "y"),
        name=aligned.name,
    )


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
        Tidy results carrying a ``radius`` column, from
        :func:`representativeness_summary` or from
        :func:`assess_representativeness` with its index reset.
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


# ------------------------------
# Top-level driver (Sects. 2.2-2.4)
# ------------------------------

#: Index levels of the tidy frame :func:`assess_representativeness` returns.
RESULT_INDEX: tuple[str, ...] = (
    "site",
    "year",
    "month",
    "period",
    "radius",
    "variable",
)

#: Aggregation level a result row describes, the ``scope`` column.
PERIOD_SCOPE: str = "period"
SITE_YEAR_SCOPE: str = "site_year"
SITE_SCOPE: str = "site"

#: Analysis a result row came from, the ``kind`` column.
CLIMATOLOGY_KIND: str = "climatology"
CATEGORICAL_KIND: str = "categorical"
CONTINUOUS_KIND: str = "continuous"

#: Value of the ``variable`` index level on rows describing the footprint
#: itself rather than a land-surface field.
FOOTPRINT_VARIABLE: str = "footprint"

#: Columns of the tidy frame, in order. Every row carries all of them; the
#: ones an analysis does not produce are missing rather than absent, so that
#: frames from different analyses concatenate without realigning.
RESULT_COLUMNS: tuple[str, ...] = (
    "scope",
    "kind",
    "time",
    "value_footprint",
    "value_target",
    "bias",
    "within_threshold",
    "dominant_class",
    "chi2",
    "dof",
    "p_value",
    "level",
    "slope",
    "slope_lower",
    "slope_upper",
    "intercept",
    "intercept_lower",
    "intercept_upper",
    "r_squared",
    "rmse",
    "mae",
    "n",
    "sufficient",
    "retained_footprint",
    "retained_target",
    "fetch",
    "area",
    "symmetry",
    "contour_level",
    "enclosed_fraction",
    "n_cells",
    "n_times",
    "seasonal_overlap",
    "daynight_overlap",
)

#: Result columns held as float64.
_FLOAT_COLUMNS: tuple[str, ...] = (
    "value_footprint",
    "value_target",
    "bias",
    "chi2",
    "p_value",
    "slope",
    "slope_lower",
    "slope_upper",
    "intercept",
    "intercept_lower",
    "intercept_upper",
    "r_squared",
    "rmse",
    "mae",
    "retained_footprint",
    "retained_target",
    "fetch",
    "area",
    "symmetry",
    "contour_level",
    "enclosed_fraction",
    "seasonal_overlap",
    "daynight_overlap",
)

#: Result columns held as pandas' nullable integer, so that a count missing
#: from an analysis stays missing rather than becoming a float.
_INT_COLUMNS: tuple[str, ...] = ("dof", "n", "n_cells", "n_times")

#: Result columns held as pandas' nullable boolean.
_BOOL_COLUMNS: tuple[str, ...] = ("within_threshold", "sufficient")


def _blank_row() -> dict[str, Any]:
    """
    Build a result row with every column present and missing.

    Returns
    -------
    dict
        Keys :data:`RESULT_INDEX` + :data:`RESULT_COLUMNS`, all set to the
        missing value of their eventual dtype. Callers overwrite the entries
        their analysis produces and leave the rest, which is what keeps the
        heterogeneous rows of the tidy frame column-aligned.
    """
    row: dict[str, Any] = {
        "site": pd.NA,
        "year": pd.NA,
        "month": pd.NaT,
        "period": pd.NA,
        "radius": float("nan"),
        "variable": pd.NA,
        "time": pd.NaT,
        "scope": pd.NA,
        "kind": pd.NA,
        "level": pd.NA,
        "dominant_class": pd.NA,
    }
    row.update({column: float("nan") for column in _FLOAT_COLUMNS})
    row.update({column: pd.NA for column in _INT_COLUMNS})
    row.update({column: pd.NA for column in _BOOL_COLUMNS})
    return row


def _result_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Assemble result rows into the tidy frame, typed and indexed.

    Parameters
    ----------
    rows : sequence of mapping
        Rows built from :func:`_blank_row`.

    Returns
    -------
    pandas.DataFrame
        Indexed by :data:`RESULT_INDEX` and carrying :data:`RESULT_COLUMNS`
        in order. An empty `rows` yields an empty frame of the same shape, so
        that a run with nothing to evaluate still concatenates.
    """
    columns = list(RESULT_INDEX) + list(RESULT_COLUMNS)
    frame = pd.DataFrame(list(rows), columns=columns)

    frame["year"] = pd.array(frame["year"], dtype="Int64")
    frame["month"] = pd.to_datetime(frame["month"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["radius"] = frame["radius"].astype(float)
    for column in _FLOAT_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in _INT_COLUMNS:
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in _BOOL_COLUMNS:
        frame[column] = pd.array(frame[column], dtype="boolean")

    return frame.set_index(list(RESULT_INDEX))[list(RESULT_COLUMNS)]


def _climatology_dataset(source: Any, **kwargs: Any) -> xr.Dataset:
    """
    Resolve the driver's first argument to a monthly-climatology Dataset.

    Parameters
    ----------
    source : xarray.Dataset, BaseFootprintModel, or xarray.DataArray
        A Dataset already carrying ``footprint_climatology`` -- the output of
        :func:`monthly_climatologies` -- is taken as it stands. Anything else
        is passed to :func:`monthly_climatologies` to be aggregated.
    **kwargs
        Forwarded to :func:`monthly_climatologies` when aggregation is needed.

    Returns
    -------
    xarray.Dataset
        Dims ``(month, period, x, y)``.

    Raises
    ------
    ValueError
        If a supplied climatology Dataset lacks the dims or coords the
        analysis indexes over.
    """
    if isinstance(source, xr.Dataset) and "footprint_climatology" in source:
        clim = source
        expected = {_MONTH_DIM, _PERIOD_DIM, "x", "y"}
        missing = expected - set(clim["footprint_climatology"].dims)
        if missing:
            raise ValueError(
                f"The climatology Dataset's 'footprint_climatology' is missing "
                f"the dimension(s) {sorted(missing)}; monthly_climatologies "
                f"returns it with dims {tuple(sorted(expected))}."
            )
        for axis in ("x", "y", _MONTH_DIM, _PERIOD_DIM):
            if axis not in clim.coords:
                raise ValueError(
                    f"The climatology Dataset carries no '{axis}' coordinate, "
                    f"so its rows cannot be labelled or placed on a grid."
                )
        return clim.transpose(_MONTH_DIM, _PERIOD_DIM, "x", "y", ...)

    return monthly_climatologies(source, **kwargs)


def _grid_axes(clim: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the tower-centred cell-centre offsets off a climatology Dataset.

    Parameters
    ----------
    clim : xarray.Dataset
        Climatology Dataset with ``x`` and ``y`` coordinates [m].

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``, one-dimensional and in metres from the tower.
    """
    return (
        np.asarray(clim.coords["x"].values, dtype=float),
        np.asarray(clim.coords["y"].values, dtype=float),
    )


def _field_on_grid(
    source: Any,
    x: np.ndarray,
    y: np.ndarray,
    station_lat: float | None,
    station_lon: float | None,
    crs: str | int | None,
    categorical: bool,
    label: str,
) -> xr.DataArray:
    """
    Bring one land-surface raster onto the tower-centred footprint grid.

    Parameters
    ----------
    source : xarray.DataArray, xarray.Dataset, str, or pathlib.Path
        An array already on the grid, a single-variable Dataset holding one,
        or a path handed to :func:`sample_raster_on_grid`.
    x, y : numpy.ndarray
        Cell-centre offsets from the tower [m].
    station_lat, station_lon : float or None
        Tower position [degrees], required only to reproject a path.
    crs : str, int, or None
        Target CRS for that reprojection.
    categorical : bool
        Whether to resample with nearest neighbour, preserving class codes.
    label : str
        How to name `source` in error messages, e.g. ``"landcover"``.

    Returns
    -------
    xarray.DataArray
        Dims ``(x, y)`` on the footprint grid.

    Raises
    ------
    TypeError
        If a Dataset holds no single usable variable.
    ValueError
        If an array is not on the footprint grid, or a path was given without
        a tower position to georeference it against.
    """
    if isinstance(source, xr.Dataset):
        names = list(source.data_vars)
        if len(names) != 1:
            raise TypeError(
                f"{label} is a Dataset holding {len(names)} variables "
                f"({', '.join(map(str, names))}); pass the one to evaluate as "
                f"a DataArray."
            )
        source = source[names[0]]

    if isinstance(source, xr.DataArray):
        field = source.transpose("x", "y") if set(source.dims) == {"x", "y"} else source
        if field.shape != (x.size, y.size):
            raise ValueError(
                f"{label} has shape {field.shape} but the footprint grid is "
                f"({x.size}, {y.size}) in (x, y). Bring it onto the grid with "
                f"sample_raster_on_grid first."
            )
        return field

    if station_lat is None or station_lon is None:
        raise ValueError(
            f"{label} was given as a path, which has to be reprojected onto "
            f"the footprint grid, but no tower position was supplied. Pass "
            f"station_lat and station_lon, or hand in a DataArray already on "
            f"the grid."
        )
    return sample_raster_on_grid(
        source, x, y, station_lat, station_lon, crs=crs, categorical=categorical
    )


def _continuous_items(continuous: Any) -> list[tuple[pd.Timestamp, Any]]:
    """
    Normalise the accepted continuous-field collections into timed entries.

    Parameters
    ----------
    continuous : mapping, pandas.Series, xarray.DataArray, xarray.Dataset, sequence, or None
        The time series of continuous rasters, in any of the forms
        :func:`assess_representativeness` documents.

    Returns
    -------
    list of tuple
        ``(timestamp, source)`` pairs in the order presented. Empty when
        `continuous` is None.

    Raises
    ------
    TypeError
        If `continuous` is of an unusable type, or a sequence entry is not a
        ``(time, raster)`` pair.
    ValueError
        If an array or Dataset carries no usable ``time`` dimension, or a
        timestamp cannot be parsed.
    """
    if continuous is None:
        return []

    if isinstance(continuous, xr.Dataset):
        names = [
            name for name in continuous.data_vars if "time" in continuous[name].dims
        ]
        if len(names) != 1:
            raise ValueError(
                f"The continuous Dataset holds {len(names)} variables with a "
                f"'time' dimension; pass the one to evaluate as a DataArray."
            )
        continuous = continuous[names[0]]

    if isinstance(continuous, xr.DataArray):
        if "time" not in continuous.dims:
            raise ValueError(
                f"The continuous DataArray has dims {continuous.dims} and so "
                f"carries no time series; pass a mapping of time -> raster "
                f"instead, or add a 'time' dimension."
            )
        stamps = pd.DatetimeIndex(continuous["time"].values)
        return [
            (stamps[index], continuous.isel(time=index, drop=True))
            for index in range(stamps.size)
        ]

    if isinstance(continuous, (Mapping, pd.Series)):
        labelled = list(continuous.items())
    else:
        try:
            labelled = [(entry[0], entry[1]) for entry in continuous]
        except (TypeError, IndexError, KeyError) as exc:
            raise TypeError(
                f"continuous must be a mapping of time -> raster, a pandas "
                f"Series of rasters indexed by time, a DataArray or Dataset "
                f"with a 'time' dimension, or an iterable of (time, raster) "
                f"pairs, got {type(continuous).__name__}."
            ) from exc

    items: list[tuple[pd.Timestamp, Any]] = []
    for label, entry in labelled:
        stamp = pd.Timestamp(label)
        if pd.isna(stamp):
            raise ValueError(
                f"The continuous field labelled {label!r} carries no usable "
                f"timestamp, so it cannot be matched to a month."
            )
        items.append((stamp, entry))
    return items


def _month_start(stamp: pd.Timestamp) -> np.datetime64:
    """
    Reduce a timestamp to the first instant of its calendar month.

    Parameters
    ----------
    stamp : pandas.Timestamp
        Any instant.

    Returns
    -------
    numpy.datetime64
        The month start, matching the ``month`` coordinate that
        :func:`monthly_climatologies` writes.
    """
    return np.datetime64(stamp.to_period("M").to_timestamp(), "ns")


def _pooled_climatology(
    weights: xr.DataArray,
    indices: Sequence[int],
) -> xr.DataArray | None:
    """
    Average a period's monthly climatologies into one pooled footprint.

    Chu et al. (2021) take the dominant land-cover type from "all available
    monthly footprint climatologies" of a site rather than from any single
    month, which is what this pooled field stands for.

    Parameters
    ----------
    weights : xarray.DataArray
        Monthly climatologies for one period, dims ``(month, x, y)``, each
        month summing to 1.
    indices : sequence of int
        Positions along ``month`` that were actually aggregated.

    Returns
    -------
    xarray.DataArray or None
        The mean of the selected months, rescaled to sum to 1, with dims
        ``(x, y)``. ``None`` when `indices` is empty or the mean carries no
        positive weight.

    Notes
    -----
    Months are weighted equally, not by their contributing half-hours: the
    paper's climatologies are the unit of analysis, and a sparse month is one
    of them. The result is a mean of unit-sum fields and so already sums to
    one; it is rescaled anyway to absorb rounding.
    """
    if not len(indices):
        return None
    pooled = weights.isel({_MONTH_DIM: list(indices)}).mean(dim=_MONTH_DIM)
    total = float(pooled.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return pooled / total


def _categorical_rows(
    w: xr.DataArray,
    landcover: xr.DataArray,
    x: np.ndarray,
    y: np.ndarray,
    radii: Sequence[float],
    alpha: float,
    template: Mapping[str, Any],
    scope: str,
    variable: str,
    label: str,
) -> list[dict[str, Any]]:
    """
    Turn one categorical evaluation into result rows, one per radius.

    Parameters
    ----------
    w : xarray.DataArray
        Footprint weights with dims ``(x, y)``, summing to 1.
    landcover : xarray.DataArray
        Categorical raster on the same grid.
    x, y : numpy.ndarray
        Cell-centre offsets from the tower [m].
    radii : sequence of float
        Target-area radii [m].
    alpha : float
        Significance level for the chi-square test.
    template : mapping
        A :func:`_blank_row` with the index levels above ``radius`` filled in.
    scope : str
        Aggregation level to stamp on the rows.
    variable : str
        Name of the land-cover product, written to the ``variable`` level.
    label : str
        Prefix for error messages, identifying the climatology evaluated.

    Returns
    -------
    list of dict
        One row per radius.

    Raises
    ------
    ValueError
        As :func:`categorical_representativeness` does, with `label` prepended.
    """
    try:
        frame = categorical_representativeness(
            w, landcover, x, y, radii=radii, alpha=alpha
        )
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc

    composition = frame.attrs.get("footprint_composition")
    retained = (
        float(composition.attrs.get("retained_weight", float("nan")))
        if composition is not None
        else float("nan")
    )

    rows: list[dict[str, Any]] = []
    for record in frame.itertuples(index=False):
        dominant = record.dominant_class
        row = dict(template)
        row.update(
            radius=float(record.radius),
            variable=variable,
            scope=scope,
            kind=CATEGORICAL_KIND,
            dominant_class=(
                int(dominant) if isinstance(dominant, (int, np.integer)) else dominant
            ),
            value_footprint=float(record.p_footprint),
            value_target=float(record.p_target),
            bias=_relative_bias(record.p_footprint, record.p_target),
            chi2=float(record.chi2),
            dof=int(record.dof),
            p_value=float(record.p_value),
            level=record.level,
            retained_footprint=retained,
        )
        rows.append(row)
    return rows


def _continuous_rows(
    w: xr.DataArray,
    field: xr.DataArray,
    x: np.ndarray,
    y: np.ndarray,
    radii: Sequence[float],
    bias_threshold: float,
    template: Mapping[str, Any],
    variable: str,
    stamp: pd.Timestamp,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Turn one matched climatology / field pair into result rows and fit pairs.

    Computes Eq. 5 once and Eq. 6 per radius, exactly as
    :func:`sensor_location_bias` does, and additionally carries the retained
    weight of both sides so that partial raster coverage stays visible.

    Parameters
    ----------
    w : xarray.DataArray
        Footprint weights with dims ``(x, y)``, summing to 1.
    field : xarray.DataArray
        Continuous field on the same grid.
    x, y : numpy.ndarray
        Cell-centre offsets from the tower [m].
    radii : sequence of float
        Target-area radii [m].
    bias_threshold : float
        ``|Delta|`` at or below which the period counts as representative.
    template : mapping
        A :func:`_blank_row` with the index levels above ``radius`` filled in.
    variable : str
        Name of the continuous field, written to the ``variable`` level.
    stamp : pandas.Timestamp
        Retrieval time of `field`, written to the ``time`` column.
    label : str
        Prefix for error messages.

    Returns
    -------
    rows : list of dict
        One result row per radius.
    pairs : list of dict
        The same matches in the four columns
        :func:`continuous_representativeness` consumes.

    Raises
    ------
    ValueError
        As :func:`footprint_weighted_value` and :func:`target_area_value` do,
        with `label` prepended.
    """
    try:
        footprint = footprint_weighted_value(w, field)
        targets = [(radius, target_area_value(field, x, y, radius)) for radius in radii]
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for radius, target in targets:
        delta = _relative_bias(footprint.value, target.value)
        row = dict(template)
        row.update(
            radius=float(radius),
            variable=variable,
            scope=PERIOD_SCOPE,
            kind=CONTINUOUS_KIND,
            time=stamp,
            value_footprint=footprint.value,
            value_target=target.value,
            bias=delta,
            within_threshold=(
                bool(abs(delta) <= float(bias_threshold))
                if np.isfinite(delta)
                else pd.NA
            ),
            retained_footprint=footprint.retained_weight,
            retained_target=target.retained_weight,
            n_cells=int(target.n_cells),
        )
        rows.append(row)
        pairs.append(
            {
                "time": stamp,
                "radius": float(radius),
                "value_footprint": footprint.value,
                "value_target": target.value,
            }
        )
    return rows, pairs


def assess_representativeness(
    model_or_climatology: Any,
    station_lat: float | None = None,
    station_lon: float | None = None,
    site_id: str | None = None,
    landcover: Any = None,
    continuous: Any = None,
    radii: Sequence[float] = TARGET_RADII,
    landcover_name: str = "land_cover",
    continuous_name: str = "EVI",
    crs: str | int | None = "auto",
    fraction: float = DEFAULT_CONTOUR_FRACTION,
    alpha: float = DEFAULT_ALPHA,
    bias_threshold: float = BIAS_THRESHOLD,
    min_matches: int = MIN_MATCHES,
    partition: str = "month+daynight",
    tz: str | float | dt.tzinfo | None = None,
    daytime: pd.Series | np.ndarray | xr.DataArray | None = None,
    min_times: int = 1,
    sw_in_pot: str | None = SW_IN_POT_COLUMN,
    ci_level: float = 0.95,
    ci_method: str = "analytical",
) -> pd.DataFrame:
    """
    Run the whole footprint-to-target-area analysis and return one tidy table.

    The driver over the rest of this module: it aggregates monthly footprint
    climatologies (Sect. 2.2), summarises each with the metrics of Eqs. 1-3,
    compares them against a series of target-area discs for a categorical and
    a continuous land-surface field (Sect. 2.4), and classifies every
    comparison HIGH, MEDIUM, or LOW. Either field may be omitted, so the same
    call serves a land-cover-only, a vegetation-index-only, or a full
    analysis.

    Parameters
    ----------
    model_or_climatology : BaseFootprintModel, xarray.Dataset, or xarray.DataArray
        Either a run footprint model or its time-resolved footprints, which
        are aggregated here with :func:`monthly_climatologies`, or a Dataset
        that function has already produced, which is used as it stands. The
        second form is the one to reach for when several analyses share one
        set of climatologies, since the aggregation is by far the slowest step.
    station_lat, station_lon : float, optional
        Tower position in decimal degrees (WGS 84). Needed to split day from
        night when the climatologies have to be aggregated here without a
        precomputed `daytime` or potential-radiation column, and to
        georeference any raster passed as a path.
    site_id : str, optional
        Site identifier, e.g. ``"US-MOz"``, written to the ``site`` index
        level so that frames from several sites concatenate. Missing when
        omitted.
    landcover : xarray.DataArray, xarray.Dataset, str, or pathlib.Path, optional
        Categorical land-cover raster -- the consolidated NLCD / Land Cover of
        Canada codes of Table S6 in the paper. A path is reprojected onto the
        footprint grid with :func:`sample_raster_on_grid` using nearest
        neighbour; a DataArray must already be on that grid. Omit to skip the
        categorical analysis.
    continuous : mapping, pandas.Series, xarray.DataArray, xarray.Dataset, or sequence, optional
        Time series of continuous rasters -- the Landsat EVI scenes of the
        paper -- in any of

        * a mapping of retrieval time -> raster (a DataArray or a path);
        * a :class:`pandas.Series` of rasters indexed by retrieval time;
        * a :class:`~xarray.DataArray` with dims ``(time, x, y)``, or a
          Dataset holding exactly one such variable;
        * an iterable of ``(time, raster)`` pairs.

        Each scene is matched to the climatologies of the calendar month it
        was retrieved in, as in Sect. 2.4. Omit to skip the continuous
        analysis.
    radii : sequence of float, default TARGET_RADII
        Target-area radii [m], evaluated and reported in the order given.
    landcover_name, continuous_name : str, default "land_cover", "EVI"
        Labels written to the ``variable`` index level, so that a frame can
        carry more than one field.
    crs : str, int, or None, default "auto"
        Target CRS used when reprojecting a raster given as a path;
        ``"auto"`` picks the local WGS 84 UTM zone.
    fraction : float, default 0.8
        Source-weight fraction each climatology is truncated at. Ignored when
        `model_or_climatology` is an already-aggregated Dataset, whose own
        fraction is used.
    alpha : float, default 0.05
        Significance level for the chi-square test and the regression.
    bias_threshold : float, default 0.10
        ``|Delta|`` at or below which a period counts as representative,
        reported as ``within_threshold``.
    min_matches : int, default 6
        Fewest matched scenes a site-level regression may be fitted on.
    partition : {"month+daynight", "month"}, default "month+daynight"
        How to group timesteps when aggregating, passed to
        :func:`monthly_climatologies`.
    tz : str, float, datetime.tzinfo, or None, optional
        Time zone of the footprint timestamps, for the day-night split.
    daytime : pandas.Series, numpy.ndarray, or xarray.DataArray, optional
        A precomputed day-night flag, used instead of computing one.
    min_times : int, default 1
        Fewest contributing timesteps a month-period group needs before it is
        aggregated and evaluated.
    sw_in_pot : str or None, default "SW_IN_POT"
        Precomputed potential-radiation column to prefer over solar geometry.
    ci_level : float, default 0.95
        Confidence level of the regression interval limits.
    ci_method : {'analytical', 'bootstrap'}, default 'analytical'
        Interval estimator, passed through to :func:`rma_regression`.

    Returns
    -------
    pandas.DataFrame
        A tidy frame indexed by :data:`RESULT_INDEX` --
        ``(site, year, month, period, radius, variable)`` -- carrying
        :data:`RESULT_COLUMNS`. Rows are of three scopes, told apart by the
        ``scope`` column, and of three kinds, told apart by ``kind``:

        ``scope="period"``, ``kind="climatology"``
            One row per aggregated month and period, carrying `fetch`,
            `area`, `symmetry`, `contour_level`, `n_cells`, and `n_times`.
            ``radius`` is missing and ``variable`` is ``"footprint"``.
        ``scope="period"``, ``kind="categorical"``
            One row per month, period, and radius: `dominant_class`,
            `value_footprint` and `value_target` (P_footprint and P_target as
            fractions), `chi2`, `dof`, `p_value`, and `level`.
        ``scope="period"``, ``kind="continuous"``
            One row per matched scene, period, and radius: `time`,
            `value_footprint` (Eq. 5), `value_target`, `bias` (Delta, Eq. 6,
            as a fraction), and `within_threshold`. These rows are the paper's
            Dataset S5.
        ``scope="site_year"``, ``kind="climatology"``
            One row per year and period, with `fetch`, `area`, and `symmetry`
            averaged over that year's months as in Fig. 3, and the overlap
            indices `seasonal_overlap` (Eq. 2) and `daynight_overlap` (Eq. 3).
            ``month`` is missing.
        ``scope="site"``, ``kind="categorical"``
            One row per period and radius, from the climatologies of every
            month pooled -- the paper's Dataset S4, and the level reported in
            its Fig. 5. ``year`` and ``month`` are missing.
        ``scope="site"``, ``kind="continuous"``
            One row per period and radius, holding the site-level model II
            regression of Eq. 7: `slope`, `intercept`, their confidence
            limits, `r_squared`, `p_value`, `rmse`, `mae`, `n`, `sufficient`,
            and `level`. The paper's Dataset S6 and Table 1.

        Columns an analysis does not produce are missing, so the frame is
        sparse by construction; select a slice with the ``scope`` and ``kind``
        columns before reading values off it. ``DataFrame.attrs`` records the
        settings the run used, including ``unmatched_fields``, the retrieval
        times of continuous rasters that fell in no aggregated month.

    Raises
    ------
    TypeError
        If `model_or_climatology`, `landcover`, or `continuous` is of an
        unusable type.
    ValueError
        If `radii` is empty or holds a radius that is not positive and finite;
        if a raster is not on the footprint grid, or was given as a path
        without a tower position; if the climatologies carry no aggregated
        month at all; or for any of the reasons the underlying evaluations
        raise, with the offending month and period prepended.
    ImportError
        If a raster given as a path needs ``rioxarray`` and it is not
        installed.

    See Also
    --------
    monthly_climatologies : The aggregation of Sect. 2.2 this drives.
    categorical_representativeness : The per-climatology land-cover step.
    sensor_location_bias : The per-period continuous step, Eq. 6.
    continuous_representativeness : The site-level regression, Eq. 7.
    export_representativeness_tables : Writes this frame in the published
        column schema of Datasets S4-S6.

    Notes
    -----
    The index is unique as long as at most one continuous raster is matched to
    each month, which is the paper's case -- it kept the cloud-free Landsat
    scenes overlapping each site's measurement periods, a median of 13 matches
    per site over several years. Two scenes in one month repeat the index and
    are told apart by the ``time`` column, so prefer ``.query`` or boolean
    masks over ``.loc`` on a frame that may hold them.

    The site-level rows pool every year, following the paper, whose Datasets
    S4 and S6 are site properties; only the overlap indices and the averaged
    fetch and area are site-*year* properties, and those are the
    ``scope="site_year"`` rows. Restrict the input to one year to get
    single-year site rows.

    The percentages of Sect. 3.3 come out of the per-period continuous rows
    directly::

        period = results.query("scope == 'period' and kind == 'continuous'")
        period.groupby("radius")["within_threshold"].mean()

    `bias` is filled for the categorical rows too, as the relative difference
    between the dominant class's footprint and target-area shares. Chu et al.
    (2021) apply the +/-10 % threshold only to the continuous field, so
    ``within_threshold`` is left missing there.

    The `level` column holds the *value* of a :class:`Level` -- ``"high"``,
    ``"medium"``, or ``"low"`` -- rather than the member itself, because
    pandas stores a string column as strings. Since ``Level`` is a ``str``
    enum, ``results["level"] == Level.HIGH`` and ``== "high"`` both select the
    same rows; only ``is Level.HIGH`` does not.

    Examples
    --------
    >>> from fluxfootprints import build_climatology, assess_representativeness
    >>> model = build_climatology(df)                          # doctest: +SKIP
    >>> results = assess_representativeness(                   # doctest: +SKIP
    ...     model,
    ...     station_lat=40.1,
    ...     station_lon=-111.9,
    ...     site_id="US-Utj",
    ...     landcover="nlcd_2019.tif",
    ...     continuous={"2019-07-04": "evi_20190704.tif"},
    ...     tz=-7,
    ... )
    >>> results.query("scope == 'site' and kind == 'continuous'")[
    ...     ["slope", "r_squared", "level"]
    ... ]                                                      # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Representativeness of Eddy-Covariance flux
    footprints for areas surrounding AmeriFlux sites. *Agric. For. Meteorol.*,
    **301-302**, 108350. https://doi.org/10.1016/j.agrformet.2021.108350
    """
    radii_values = [float(radius) for radius in radii]
    if not radii_values:
        raise ValueError(
            "radii holds no target areas, so there is nothing to compare the "
            "footprint against. Pass at least one radius, e.g. TARGET_RADII."
        )
    for radius in radii_values:
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"Every target radius must be positive and finite, got "
                f"{radius!r}."
            )

    clim = _climatology_dataset(
        model_or_climatology,
        partition=partition,
        latitude=station_lat,
        longitude=station_lon,
        tz=tz,
        daytime=daytime,
        fraction=fraction,
        min_times=min_times,
        sw_in_pot=sw_in_pot,
    )
    x, y = _grid_axes(clim)
    weights_all = clim["footprint_climatology"]
    months = pd.DatetimeIndex(clim[_MONTH_DIM].values)
    periods = [str(period) for period in np.atleast_1d(clim[_PERIOD_DIM].values)]
    fraction_used = float(
        weights_all.attrs.get("contour_fraction", clim.attrs.get("contour_fraction", fraction))
    )
    site = pd.NA if site_id is None else str(site_id)

    landcover_field = None
    if landcover is not None:
        landcover_field = _field_on_grid(
            landcover, x, y, station_lat, station_lon, crs, True, "landcover"
        )

    fields_by_month: dict[Any, list[tuple[pd.Timestamp, xr.DataArray]]] = {}
    for stamp, source in _continuous_items(continuous):
        field = _field_on_grid(
            source,
            x,
            y,
            station_lat,
            station_lon,
            crs,
            False,
            f"the continuous field at {stamp}",
        )
        fields_by_month.setdefault(_month_start(stamp), []).append((stamp, field))

    rows: list[dict[str, Any]] = []
    populated: dict[str, list[int]] = {period: [] for period in periods}
    pairs: dict[str, list[dict[str, Any]]] = {period: [] for period in periods}
    monthly_metrics: dict[tuple[int, str], ClimatologyMetrics] = {}
    matched_months: set[Any] = set()

    for mi, month in enumerate(months):
        month_key = np.datetime64(month.to_datetime64(), "ns")
        for pi, period in enumerate(periods):
            selector = {_MONTH_DIM: mi, _PERIOD_DIM: pi}
            n_times = int(clim["n_times"].isel(selector)) if "n_times" in clim else 1
            w = weights_all.isel(selector)
            values = np.asarray(w.values, dtype=float)
            if n_times < 1 or not np.isfinite(values).any() or np.nansum(values) <= 0.0:
                continue
            if "contour_fraction" not in w.attrs:
                w = w.assign_attrs(contour_fraction=fraction_used)

            populated[period].append(mi)
            metrics = climatology_metrics(w, fraction=fraction_used)
            monthly_metrics[(mi, period)] = metrics

            template = _blank_row()
            template.update(
                site=site, year=int(month.year), month=month, period=period
            )
            label = f"At {month:%Y-%m} {period}"

            row = dict(template)
            row.update(
                variable=FOOTPRINT_VARIABLE,
                scope=PERIOD_SCOPE,
                kind=CLIMATOLOGY_KIND,
                fetch=metrics.fetch,
                area=metrics.area,
                symmetry=metrics.symmetry,
                enclosed_fraction=metrics.enclosed_fraction,
                contour_level=(
                    float(clim["contour_level"].isel(selector))
                    if "contour_level" in clim
                    else metrics.contour_level
                ),
                n_cells=int(metrics.n_cells),
                n_times=n_times,
            )
            rows.append(row)

            if landcover_field is not None:
                rows.extend(
                    _categorical_rows(
                        w,
                        landcover_field,
                        x,
                        y,
                        radii_values,
                        alpha,
                        template,
                        PERIOD_SCOPE,
                        landcover_name,
                        label,
                    )
                )

            for stamp, field in fields_by_month.get(month_key, []):
                matched_months.add(month_key)
                scene_rows, scene_pairs = _continuous_rows(
                    w,
                    field,
                    x,
                    y,
                    radii_values,
                    bias_threshold,
                    template,
                    continuous_name,
                    stamp,
                    f"{label}, field at {stamp}",
                )
                rows.extend(scene_rows)
                pairs[period].extend(scene_pairs)

    if not any(populated.values()):
        raise ValueError(
            "No month-period group carries an aggregated climatology, so "
            "there is nothing to evaluate. Check that the footprints span at "
            "least one month and that min_times is not set above the number "
            "of available timesteps."
        )

    rows.extend(
        _site_year_rows(weights_all, months, periods, populated, monthly_metrics, site)
    )
    rows.extend(
        _site_rows(
            weights_all,
            periods,
            populated,
            pairs,
            landcover_field,
            x,
            y,
            radii_values,
            alpha,
            fraction_used,
            landcover_name,
            continuous_name,
            min_matches,
            ci_level,
            ci_method,
            site,
        )
    )

    results = _result_frame(rows)
    results.attrs.update(
        {
            "site_id": site_id,
            "contour_fraction": fraction_used,
            "radii": tuple(radii_values),
            "alpha": float(alpha),
            "bias_threshold": float(bias_threshold),
            "min_matches": int(min_matches),
            "ci_level": float(ci_level),
            "ci_method": ci_method,
            "partition": str(clim.attrs.get("partition", partition)),
            "landcover_variable": None if landcover is None else landcover_name,
            "continuous_variable": None if continuous is None else continuous_name,
            "unmatched_fields": tuple(
                str(stamp)
                for key, entries in sorted(fields_by_month.items())
                for stamp, _ in entries
                if key not in matched_months
            ),
            "reference": (
                "Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350."
            ),
        }
    )
    return results


def _site_year_rows(
    weights_all: xr.DataArray,
    months: pd.DatetimeIndex,
    periods: Sequence[str],
    populated: Mapping[str, Sequence[int]],
    monthly_metrics: Mapping[tuple[int, str], ClimatologyMetrics],
    site: Any,
) -> list[dict[str, Any]]:
    """
    Build the site-year climatology rows: averaged metrics and the overlaps.

    Parameters
    ----------
    weights_all : xarray.DataArray
        Every climatology, dims ``(month, period, x, y)``.
    months : pandas.DatetimeIndex
        The ``month`` coordinate, one entry per calendar month.
    periods : sequence of str
        The ``period`` coordinate.
    populated : mapping
        Period -> positions along ``month`` that were aggregated.
    monthly_metrics : mapping
        ``(month position, period)`` -> the metrics already computed for it.
    site : Any
        Site label for the index.

    Returns
    -------
    list of dict
        One row per year and period holding at least one aggregated month.

    Notes
    -----
    Fetch, area, and symmetry are averaged over the year's months, as Fig. 3
    of the paper does -- symmetry as the mean of the monthly indices, not as
    Eq. 1 applied to the mean area and fetch, which is not the same number.
    The seasonal overlap of Eq. 2 needs at least two months and is missing
    below that; the day-night overlap of Eq. 3 needs a month whose daytime and
    nighttime climatologies were both aggregated, and is written to both
    period rows of the year since it belongs to neither alone.
    """
    years = sorted(
        {int(months[index].year) for entries in populated.values() for index in entries}
    )

    rows: list[dict[str, Any]] = []
    for year in years:
        in_year = {
            period: [index for index in populated[period] if months[index].year == year]
            for period in periods
        }

        daynight = float("nan")
        if DAYTIME in periods and NIGHTTIME in periods:
            shared = sorted(set(in_year[DAYTIME]) & set(in_year[NIGHTTIME]))
            if shared:
                daynight = daynight_overlap(
                    weights_all.isel(
                        {_PERIOD_DIM: periods.index(DAYTIME), _MONTH_DIM: shared}
                    ),
                    weights_all.isel(
                        {_PERIOD_DIM: periods.index(NIGHTTIME), _MONTH_DIM: shared}
                    ),
                )

        for pi, period in enumerate(periods):
            indices = in_year[period]
            if not indices:
                continue
            metrics = [monthly_metrics[(index, period)] for index in indices]

            row = _blank_row()
            row.update(
                site=site,
                year=year,
                period=period,
                variable=FOOTPRINT_VARIABLE,
                scope=SITE_YEAR_SCOPE,
                kind=CLIMATOLOGY_KIND,
                fetch=float(np.mean([entry.fetch for entry in metrics])),
                area=float(np.mean([entry.area for entry in metrics])),
                symmetry=float(np.mean([entry.symmetry for entry in metrics])),
                n_cells=int(round(float(np.mean([entry.n_cells for entry in metrics])))),
                seasonal_overlap=(
                    seasonal_overlap(
                        weights_all.isel({_PERIOD_DIM: pi, _MONTH_DIM: indices})
                    )
                    if len(indices) > 1
                    else float("nan")
                ),
                daynight_overlap=daynight,
            )
            rows.append(row)
    return rows


def _site_rows(
    weights_all: xr.DataArray,
    periods: Sequence[str],
    populated: Mapping[str, Sequence[int]],
    pairs: Mapping[str, Sequence[Mapping[str, Any]]],
    landcover_field: xr.DataArray | None,
    x: np.ndarray,
    y: np.ndarray,
    radii: Sequence[float],
    alpha: float,
    fraction: float,
    landcover_name: str,
    continuous_name: str,
    min_matches: int,
    ci_level: float,
    ci_method: str,
    site: Any,
) -> list[dict[str, Any]]:
    """
    Build the site-level rows: the pooled land cover and the regression.

    Parameters
    ----------
    weights_all : xarray.DataArray
        Every climatology, dims ``(month, period, x, y)``.
    periods : sequence of str
        The ``period`` coordinate.
    populated : mapping
        Period -> positions along ``month`` that were aggregated.
    pairs : mapping
        Period -> the matched rows :func:`continuous_representativeness` takes.
    landcover_field : xarray.DataArray or None
        Categorical raster on the footprint grid, or None to skip.
    x, y : numpy.ndarray
        Cell-centre offsets from the tower [m].
    radii : sequence of float
        Target-area radii [m].
    alpha : float
        Significance level for both classifications.
    fraction : float
        Source-weight fraction the climatologies were truncated at.
    landcover_name, continuous_name : str
        Labels for the ``variable`` index level.
    min_matches : int
        Fewest matched scenes a regression may be fitted on.
    ci_level : float
        Confidence level of the interval limits.
    ci_method : str
        Interval estimator.
    site : Any
        Site label for the index.

    Returns
    -------
    list of dict
        The Dataset S4 and S6 rows, ``year`` and ``month`` missing on both.
    """
    rows: list[dict[str, Any]] = []
    for pi, period in enumerate(periods):
        indices = list(populated[period])
        if not indices:
            continue

        template = _blank_row()
        template.update(site=site, period=period)

        if landcover_field is not None:
            pooled = _pooled_climatology(weights_all.isel({_PERIOD_DIM: pi}), indices)
            if pooled is not None:
                rows.extend(
                    _categorical_rows(
                        pooled.assign_attrs(contour_fraction=fraction),
                        landcover_field,
                        x,
                        y,
                        radii,
                        alpha,
                        template,
                        SITE_SCOPE,
                        landcover_name,
                        f"Pooled over every {period} climatology",
                    )
                )

        if not pairs[period]:
            continue
        fits = continuous_representativeness(
            pd.DataFrame(list(pairs[period])),
            radii=radii,
            min_matches=min_matches,
            alpha=alpha,
            ci_level=ci_level,
            ci_method=ci_method,
        )
        for record in fits.itertuples(index=False):
            row = dict(template)
            row.update(
                radius=float(record.radius),
                variable=continuous_name,
                scope=SITE_SCOPE,
                kind=CONTINUOUS_KIND,
                slope=float(record.slope),
                slope_lower=float(record.slope_lower),
                slope_upper=float(record.slope_upper),
                intercept=float(record.intercept),
                intercept_lower=float(record.intercept_lower),
                intercept_upper=float(record.intercept_upper),
                r_squared=float(record.r_squared),
                p_value=float(record.p_value),
                rmse=float(record.rmse),
                mae=float(record.mae),
                n=int(record.n),
                sufficient=bool(record.sufficient),
                level=record.level,
            )
            rows.append(row)
    return rows


# ------------------------------
# Publication tables (Datasets S4-S6)
# ------------------------------

#: Column order of each reconstructed supplementary table, keyed by the
#: dataset number Chu et al. (2021) publish it under. See
#: :func:`representativeness_table` for what each holds.
CHU_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "S4": (
        "SITE_ID",
        "PERIOD",
        "TARGET_AREA",
        "PRODUCT",
        "DOM_LC",
        "P_FOOTPRINT",
        "P_TARGET",
        "P_DIFF",
        "CHI2",
        "DF",
        "P_VALUE",
        "REP_LC",
    ),
    "S5": (
        "SITE_ID",
        "YEAR",
        "MONTH",
        "PERIOD",
        "TARGET_AREA",
        "VARIABLE",
        "TIMESTAMP",
        "EVI_FOOTPRINT",
        "EVI_TARGET",
        "DELTA",
        "WITHIN_THRESHOLD",
    ),
    "S6": (
        "SITE_ID",
        "PERIOD",
        "TARGET_AREA",
        "VARIABLE",
        "N",
        "INTERCEPT",
        "INTERCEPT_LCL",
        "INTERCEPT_UCL",
        "SLOPE",
        "SLOPE_LCL",
        "SLOPE_UCL",
        "R2",
        "RMSE",
        "MAE",
        "REP_EVI",
    ),
}

#: File formats :func:`export_representativeness_tables` writes.
_TABLE_FORMATS: dict[str, str] = {"csv": ".csv", "parquet": ".parquet"}


def _table_s4(results: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the site-level land-cover rows into the Dataset S4 layout.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`assess_representativeness`, index reset.

    Returns
    -------
    pandas.DataFrame
        Columns ``CHU_TABLE_COLUMNS["S4"]``, with the shares as percentages.
    """
    rows = results[
        (results["scope"] == SITE_SCOPE) & (results["kind"] == CATEGORICAL_KIND)
    ]
    table = pd.DataFrame(index=rows.index)
    table["SITE_ID"] = rows["site"]
    table["PERIOD"] = rows["period"]
    table["TARGET_AREA"] = rows["radius"]
    table["PRODUCT"] = rows["variable"]
    table["DOM_LC"] = rows["dominant_class"]
    table["P_FOOTPRINT"] = 100.0 * rows["value_footprint"]
    table["P_TARGET"] = 100.0 * rows["value_target"]
    table["P_DIFF"] = table["P_FOOTPRINT"] - table["P_TARGET"]
    table["CHI2"] = rows["chi2"]
    table["DF"] = rows["dof"]
    table["P_VALUE"] = rows["p_value"]
    table["REP_LC"] = rows["level"].map(_level_label)
    return table.reset_index(drop=True)


def _table_s5(results: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the per-period continuous rows into the Dataset S5 layout.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`assess_representativeness`, index reset.

    Returns
    -------
    pandas.DataFrame
        Columns ``CHU_TABLE_COLUMNS["S5"]``, with the bias as a percentage
        and the month as its number in the year.
    """
    rows = results[
        (results["scope"] == PERIOD_SCOPE) & (results["kind"] == CONTINUOUS_KIND)
    ]
    table = pd.DataFrame(index=rows.index)
    table["SITE_ID"] = rows["site"]
    table["YEAR"] = rows["year"]
    table["MONTH"] = pd.to_datetime(rows["month"]).dt.month.astype("Int64")
    table["PERIOD"] = rows["period"]
    table["TARGET_AREA"] = rows["radius"]
    table["VARIABLE"] = rows["variable"]
    table["TIMESTAMP"] = rows["time"]
    table["EVI_FOOTPRINT"] = rows["value_footprint"]
    table["EVI_TARGET"] = rows["value_target"]
    table["DELTA"] = 100.0 * rows["bias"]
    table["WITHIN_THRESHOLD"] = rows["within_threshold"]
    return table.reset_index(drop=True)


def _table_s6(results: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the site-level regression rows into the Dataset S6 layout.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`assess_representativeness`, index reset.

    Returns
    -------
    pandas.DataFrame
        Columns ``CHU_TABLE_COLUMNS["S6"]``, in the order of Table 1.
    """
    rows = results[
        (results["scope"] == SITE_SCOPE) & (results["kind"] == CONTINUOUS_KIND)
    ]
    table = pd.DataFrame(index=rows.index)
    table["SITE_ID"] = rows["site"]
    table["PERIOD"] = rows["period"]
    table["TARGET_AREA"] = rows["radius"]
    table["VARIABLE"] = rows["variable"]
    table["N"] = rows["n"]
    table["INTERCEPT"] = rows["intercept"]
    table["INTERCEPT_LCL"] = rows["intercept_lower"]
    table["INTERCEPT_UCL"] = rows["intercept_upper"]
    table["SLOPE"] = rows["slope"]
    table["SLOPE_LCL"] = rows["slope_lower"]
    table["SLOPE_UCL"] = rows["slope_upper"]
    table["R2"] = rows["r_squared"]
    table["RMSE"] = rows["rmse"]
    table["MAE"] = rows["mae"]
    table["REP_EVI"] = rows["level"].map(_level_label)
    return table.reset_index(drop=True)


#: The reshapers, keyed by dataset number.
_TABLE_BUILDERS = {"S4": _table_s4, "S5": _table_s5, "S6": _table_s6}


def _level_label(level: Any) -> Any:
    """
    Render a representativeness level as the plain string a table carries.

    Parameters
    ----------
    level : Level, str, or pandas.NA
        A cell of the ``level`` column.

    Returns
    -------
    str or pandas.NA
        ``"high"``, ``"medium"``, ``"low"``, or the missing value unchanged.
    """
    if isinstance(level, Level):
        return level.value
    return level if isinstance(level, str) else pd.NA


def representativeness_table(results: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Reshape a tidy result frame into one of the published table layouts.

    Chu et al. (2021) release their per-site statistics as Datasets S4-S6 on
    Zenodo (doi:10.5281/zenodo.4015350). This puts the corresponding rows of
    :func:`assess_representativeness` into the same shape and units, so that a
    local analysis can be read next to, or concatenated with, the published
    tables.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`assess_representativeness`, indexed or with the index
        already reset.
    dataset : {"S4", "S5", "S6"}
        Which table to build, case-insensitively:

        ``"S4"``
            Land-cover statistics: the dominant class, P_footprint, P_target,
            their difference, and the chi-square test, per period and target
            area. Built from the ``scope="site"`` categorical rows.
        ``"S5"``
            Site-month vegetation-index statistics: the footprint-weighted and
            target-area values and the sensor location bias Delta, per matched
            scene, period, and target area. Built from the ``scope="period"``
            continuous rows.
        ``"S6"``
            Site-level model II regressions in the layout of Table 1:
            intercept and slope with their confidence limits, R², RMSE, and
            MAE. Built from the ``scope="site"`` continuous rows.

    Returns
    -------
    pandas.DataFrame
        The requested table over a fresh :class:`~pandas.RangeIndex`, with the
        columns of ``CHU_TABLE_COLUMNS[dataset]`` in that order. Empty, but
        still carrying every column, when `results` holds no rows of that
        kind -- an analysis run without a land-cover raster has an empty S4.

    Raises
    ------
    KeyError
        If `dataset` is not one of the three.
    ValueError
        If `results` is missing the columns of a tidy result frame.

    See Also
    --------
    assess_representativeness : Produces the frame this reshapes.
    export_representativeness_tables : Writes these tables to disk.

    Notes
    -----
    The shares of S4 and the bias of S5 are converted to the percentages the
    paper reports, while the tidy frame keeps them as fractions like the rest
    of this module. Units are the only transformation: no row is dropped,
    reordered, or recomputed.

    The column *names* are reconstructed from the paper's text -- Sects. 2.4
    and 3.3, and Table 1 -- rather than copied from the Zenodo archive, which
    this package does not ship or read. Treat them as a faithful layout rather
    than a byte-for-byte match, and check against the archive before merging
    on column names. Two columns have no counterpart in the paper's own tables
    and are added because a local analysis needs them: ``PRODUCT`` and
    ``VARIABLE``, naming the raster each row came from, and ``TIMESTAMP``, the
    retrieval time of the matched scene, which the paper resolves only to the
    month.

    Examples
    --------
    >>> table = representativeness_table(results, "S6")   # doctest: +SKIP
    >>> table[["TARGET_AREA", "SLOPE", "R2", "REP_EVI"]]  # doctest: +SKIP
    """
    key = str(dataset).upper()
    if key not in _TABLE_BUILDERS:
        raise KeyError(
            f"dataset must be one of {sorted(_TABLE_BUILDERS)}, got "
            f"{dataset!r}."
        )

    flat = results.reset_index() if results.index.nlevels > 1 else results.copy()
    missing = [
        column
        for column in list(RESULT_INDEX) + ["scope", "kind"]
        if column not in flat.columns
    ]
    if missing:
        raise ValueError(
            f"results is missing the column(s) {missing}; pass the frame "
            f"assess_representativeness returns, indexed or reset."
        )

    table = _TABLE_BUILDERS[key](flat)
    return table.reindex(columns=list(CHU_TABLE_COLUMNS[key]))


def export_representativeness_tables(
    results: pd.DataFrame,
    directory: str | Path,
    datasets: Sequence[str] = ("S4", "S5", "S6"),
    fmt: str = "csv",
    prefix: str = "representativeness",
    write_empty: bool = False,
    **kwargs: Any,
) -> dict[str, Path]:
    """
    Write the published-schema tables of a result frame to CSV or Parquet.

    A thin writer over :func:`representativeness_table`: it builds each
    requested table and writes it as ``<prefix>_<dataset>.<ext>``, so that the
    outputs sit next to Chu et al.'s Datasets S4-S6 (Zenodo,
    doi:10.5281/zenodo.4015350) with matching columns and units.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`assess_representativeness`. Frames from several sites
        may be concatenated first; the ``SITE_ID`` column keeps them apart.
    directory : str or pathlib.Path
        Directory to write into. Created, parents included, if it does not
        exist.
    datasets : sequence of str, default ("S4", "S5", "S6")
        Which tables to write, from ``"S4"``, ``"S5"``, and ``"S6"``.
    fmt : {"csv", "parquet"}, default "csv"
        Output format. Parquet needs ``pyarrow`` or ``fastparquet``, and
        preserves the nullable integer and boolean dtypes that CSV flattens
        to text.
    prefix : str, default "representativeness"
        Leading part of each file name.
    write_empty : bool, default False
        Whether to write a header-only file for a table with no rows -- an S4
        from a run without a land-cover raster, say. When False such a table
        is skipped and simply does not appear in the returned mapping.
    **kwargs
        Passed through to :meth:`pandas.DataFrame.to_csv` or
        :meth:`~pandas.DataFrame.to_parquet`. ``index=False`` is set for CSV
        unless overridden.

    Returns
    -------
    dict of str to pathlib.Path
        Dataset number -> the path written, in the order requested. Datasets
        skipped as empty are absent.

    Raises
    ------
    KeyError
        If `datasets` names a table that is not one of the three.
    ValueError
        If `fmt` is neither ``"csv"`` nor ``"parquet"``, or if `results` is
        not a tidy result frame.
    ImportError
        If Parquet was asked for and no Parquet engine is installed.

    See Also
    --------
    representativeness_table : Builds one table without writing it.
    assess_representativeness : Produces the frame this writes.
    export_representativeness_gpkg : Writes the same results as target-area
        discs for mapping.

    Notes
    -----
    The tidy frame itself is the thing to keep for further analysis -- these
    tables drop the climatology metrics and the per-month land-cover rows,
    which have no counterpart in the published supplement. Write it alongside
    with ``results.to_parquet(...)`` when the full record matters.

    Examples
    --------
    >>> written = export_representativeness_tables(      # doctest: +SKIP
    ...     results, "out/", fmt="parquet", prefix="US-Utj"
    ... )
    >>> sorted(written)                                  # doctest: +SKIP
    ['S4', 'S5', 'S6']
    """
    if fmt not in _TABLE_FORMATS:
        raise ValueError(
            f"fmt must be one of {sorted(_TABLE_FORMATS)}, got {fmt!r}."
        )

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for dataset in datasets:
        key = str(dataset).upper()
        table = representativeness_table(results, key)
        if table.empty and not write_empty:
            continue

        path = target / f"{prefix}_{key}{_TABLE_FORMATS[fmt]}"
        if fmt == "csv":
            table.to_csv(path, **{"index": False, **kwargs})
        else:
            table.to_parquet(path, **kwargs)
        written[key] = path
    return written
