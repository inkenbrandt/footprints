# src/fluxfootprints/representativeness_data.py
"""
representativeness_data.py
==========================
Google Earth Engine retrieval of the surface fields Chu et al. (2021) evaluate.

The representativeness analysis in :mod:`~fluxfootprints.representativeness`
compares a footprint climatology against two external surface fields: a
land-cover map for the categorical evaluation of Sect. 2.4, and a Landsat
vegetation index for the continuous one. This module fetches both straight from
Earth Engine onto a tower-centred grid, so a site can be assessed without first
downloading and clipping national mosaics by hand:

    Chu, H., et al. (2021). Representativeness of Eddy-Covariance flux
    footprints for areas surrounding AmeriFlux sites. *Agricultural and Forest
    Meteorology*, **301-302**, 108350.
    https://doi.org/10.1016/j.agrformet.2021.108350

========================== ==============================================
Function                   Field
========================== ==============================================
:func:`fetch_nlcd`         NLCD land cover, for the categorical evaluation
:func:`fetch_landsat_evi`  Landsat EVI (Eq. 4), for the continuous one
========================== ==============================================

Both return :class:`xarray.DataArray` on a north-up grid in the CRS they were
asked for, georeferenced with ``.rio.write_crs()`` and carrying ``nan`` wherever
there is no usable observation -- which is what
``representativeness._align_raster`` expects of a source raster, so the result
goes to :func:`~fluxfootprints.evaluate_landcover` or
:func:`~fluxfootprints.evaluate_vegetation_index` after a single warp onto the
footprint grid.

Optional dependency
-------------------
This module is import-guarded: ``earthengine-api`` is resolved lazily inside
:func:`initialize`, so importing :mod:`fluxfootprints` -- or this module --
never requires it. Only calling a fetch function does::

    pip install 'fluxfootprints[gee]'

Earth Engine also needs a one-off authentication and a Cloud project::

    earthengine authenticate

after which :func:`initialize`, called for you by both fetch functions, picks
the stored credentials up.

Notes
-----
The fetch functions issue network requests, so nothing here is exercised
offline beyond a stubbed ``ee``; treat their output as you would any other
remotely sourced raster and check the coverage each one reports before leaning
on it.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

from .representativeness import _require

__all__ = [
    "NLCD_COLLECTION",
    "NLCD_BAND",
    "LANDSAT_COLLECTIONS",
    "LandsatBands",
    "SR_SCALE",
    "SR_OFFSET",
    "DEFAULT_SCALE",
    "DEFAULT_MAX_CLOUD",
    "DEFAULT_MAX_SCENES",
    "initialize",
    "fetch_nlcd",
    "fetch_landsat_evi",
]


# ------------------------------
# Constants
# ------------------------------

#: Earth Engine id of the land-cover collection :func:`fetch_nlcd` reads.
#: Chu et al. (2021) used NLCD 2016, together with the Canadian Land Cover map
#: for their Canadian sites, which this module does not cover.
NLCD_COLLECTION: str = "USGS/NLCD_RELEASES/2019_REL/NLCD"

#: Band of :data:`NLCD_COLLECTION` holding the Anderson class codes.
NLCD_BAND: str = "landcover"


@dataclass(frozen=True)
class LandsatBands:
    """
    Band names of one Landsat sensor's Collection 2 surface-reflectance product.

    Attributes
    ----------
    collection : str
        Earth Engine ImageCollection id.
    blue, red, nir : str
        Bands entering the EVI of Eq. 4, in that role.
    qa : str
        Band holding the CFMask quality bits, ``QA_PIXEL`` throughout
        Collection 2.
    label : str
        Human-readable sensor name, recorded on the returned array.
    """

    collection: str
    blue: str
    red: str
    nir: str
    qa: str
    label: str


#: Sensors :func:`fetch_landsat_evi` draws on. Landsat 7 / ETM+ is deliberately
#: absent: Chu et al. (2021) skip it, as the scan-line corrector failure of
#: 2003 leaves wedge-shaped gaps a footprint-weighted mean cannot be taken
#: across.
LANDSAT_COLLECTIONS: tuple[LandsatBands, ...] = (
    LandsatBands(
        collection="LANDSAT/LT05/C02/T1_L2",
        blue="SR_B1",
        red="SR_B3",
        nir="SR_B4",
        qa="QA_PIXEL",
        label="LANDSAT_5",
    ),
    LandsatBands(
        collection="LANDSAT/LC08/C02/T1_L2",
        blue="SR_B2",
        red="SR_B4",
        nir="SR_B5",
        qa="QA_PIXEL",
        label="LANDSAT_8",
    ),
)

#: Scale factor of the Collection 2 Level-2 surface-reflectance bands, which
#: are stored as unsigned 16-bit integers.
SR_SCALE: float = 2.75e-05

#: Additive offset of those bands, applied after :data:`SR_SCALE`.
SR_OFFSET: float = -0.2

#: Native ground sample distance of both NLCD and Landsat [m].
DEFAULT_SCALE: float = 30.0

#: Fraction of the target disc allowed to lack a clear observation before a
#: Landsat scene is rejected; the "< 1 % cloud" criterion of Chu et al. (2021).
DEFAULT_MAX_CLOUD: float = 0.01

#: Scenes :func:`fetch_landsat_evi` will pull in one call before refusing.
#: Chu et al. (2021) matched 1-103 scenes per site, median 13.
DEFAULT_MAX_SCENES: int = 200

#: ``QA_PIXEL`` bits marking a pixel as carrying no clear surface view:
#: dilated cloud (1), cirrus (2, OLI only), cloud (3), cloud shadow (4).
_QA_OBSCURED_BITS: tuple[int, ...] = (1, 2, 3, 4)

#: ``QA_PIXEL`` bit marking designated fill, i.e. a pixel inside the scene
#: bounding box but outside the imaged swath.
_QA_FILL_BIT: int = 0

#: Documented response cap of ``ee.data.computePixels`` [bytes]. Requests are
#: sized against it up front, so an over-large grid fails with a message naming
#: the knobs rather than with an opaque server error.
_MAX_RESPONSE_BYTES: int = 48 * 1024 * 1024


# ------------------------------
# Earth Engine session
# ------------------------------

#: Set once :func:`initialize` has driven ``ee.Initialize`` to completion, so
#: repeated fetches in one process do not re-enter it.
_INITIALIZED: bool = False


def initialize(project: str | None = None, **kwargs: Any) -> ModuleType:
    """
    Import and initialise the Earth Engine client, at most once per process.

    Both fetch functions call this themselves, so it is worth calling directly
    only to choose the Cloud project, or to fail early -- before a long
    analysis -- if the credentials are not in place.

    Parameters
    ----------
    project : str, optional
        Google Cloud project to bill the requests to. Earth Engine requires one
        unless a default is already stored in the local credentials.
    **kwargs
        Further keyword arguments for ``ee.Initialize``, e.g. ``credentials``
        or ``opt_url``.

    Returns
    -------
    types.ModuleType
        The imported ``ee`` module, ready to use.

    Raises
    ------
    ImportError
        If ``earthengine-api`` is not installed, with the install command in
        the message.
    RuntimeError
        If ``ee.Initialize`` fails, with the authentication command in the
        message.

    Notes
    -----
    A session the caller has already initialised is left alone: if ``ee`` is
    holding credentials this returns the module untouched, so an application
    that initialises Earth Engine its own way keeps its settings.

    Examples
    --------
    >>> ee = initialize(project="my-cloud-project")   # doctest: +SKIP
    """
    global _INITIALIZED

    ee = _require("ee")
    if _INITIALIZED or getattr(ee.data, "_credentials", None) is not None:
        _INITIALIZED = True
        return ee

    try:
        ee.Initialize(project=project, **kwargs)
    except Exception as exc:  # pragma: no cover - needs a live EE session
        raise RuntimeError(
            "Earth Engine could not be initialised. Authenticate once with\n"
            "    earthengine authenticate\n"
            "and pass a Cloud project, e.g. "
            "initialize(project='my-cloud-project')."
        ) from exc

    _INITIALIZED = True
    return ee


# ------------------------------
# Grid construction
# ------------------------------


def _crs_spec(crs: str | int | CRS) -> tuple[CRS, dict[str, str]]:
    """
    Resolve a user-supplied CRS into a parsed CRS and an Earth Engine grid key.

    Parameters
    ----------
    crs : str, int, or pyproj.CRS
        Anything :meth:`pyproj.CRS.from_user_input` accepts, e.g. an EPSG code,
        a WKT string, or the ``crs`` of a
        :class:`~fluxfootprints.GridGeometry`.

    Returns
    -------
    parsed : pyproj.CRS
        The CRS as pyproj understands it.
    spec : dict
        Either ``{"crsCode": "EPSG:nnnn"}`` or, for a CRS carrying no authority
        code, ``{"crsWkt": ...}``; ready to merge into a ``PixelGrid``.

    Raises
    ------
    ValueError
        If `crs` cannot be parsed, or is geographic rather than projected --
        `radius` and `scale` are metres, and the footprint grid this feeds is
        metres too.
    """
    try:
        parsed = CRS.from_user_input(crs)
    except Exception as exc:
        raise ValueError(
            f"crs={crs!r} could not be interpreted as a coordinate reference "
            "system; pass an EPSG code, a WKT string, or a pyproj.CRS."
        ) from exc

    if parsed.is_geographic:
        raise ValueError(
            f"crs must be a projected CRS in metres, but {parsed.name!r} is "
            "geographic (degrees); radius and scale are metres, and the "
            "footprint grid this feeds is metric too. Use the local UTM zone, "
            "e.g. the crs of footprint_grid_geometry(..., crs='auto')."
        )

    epsg = parsed.to_epsg()
    if epsg is not None:
        return parsed, {"crsCode": f"EPSG:{epsg}"}
    return parsed, {"crsWkt": parsed.to_wkt()}


def _projection(spec: dict[str, str]) -> str:
    """
    The projection string Earth Engine geometry constructors take.

    Parameters
    ----------
    spec : dict
        CRS key from :func:`_crs_spec`.

    Returns
    -------
    str
        The authority code where there is one, else the WKT.
    """
    return spec.get("crsCode") or spec["crsWkt"]


def _pixel_grid(
    x: float,
    y: float,
    radius: float,
    scale: float,
    spec: dict[str, str],
    bands: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """
    Build the ``PixelGrid`` of a tower-centred square, and its coordinate axes.

    Parameters
    ----------
    x, y : float
        Tower position in the target CRS [m].
    radius : float
        Half-width of the square requested [m]. It circumscribes the target
        disc of the same radius, so the result covers every target area out to
        `radius`.
    scale : float
        Cell size [m].
    spec : dict
        CRS key from :func:`_crs_spec`.
    bands : int
        Number of float64 bands the request will return, used only to size the
        response against :data:`_MAX_RESPONSE_BYTES`.

    Returns
    -------
    grid : dict
        ``PixelGrid`` for ``ee.data.computePixels``.
    xs, ys : numpy.ndarray
        Cell-centre coordinates in the target CRS [m], ``ys`` descending as a
        north-up raster requires.

    Raises
    ------
    ValueError
        If the tower position is not finite, if `radius` or `scale` is not
        positive, or if the grid the two imply would exceed the Earth Engine
        response cap.

    Notes
    -----
    The square is centred exactly on the tower rather than snapped to a
    multiple of `scale`, so it is reproducible from ``(x, y, radius, scale)``
    alone and is symmetric about the tower the way the target discs are. The
    cell count is rounded up, which can push the edges a fraction of a cell
    past `radius` -- harmless, since this array is the source of a warp and not
    the analysis grid itself.
    """
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"the tower position must be finite, got x={x!r}, y={y!r}.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            f"radius must be a positive distance in metres, got {radius!r}."
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"scale must be a positive cell size in metres, got {scale!r}."
        )

    width = height = math.ceil(2.0 * radius / scale)
    span = width * scale / 2.0

    estimate = width * height * bands * 8
    if estimate > _MAX_RESPONSE_BYTES:
        raise ValueError(
            f"a {width} x {height} grid of {bands} band(s) is about "
            f"{estimate / 1024 ** 2:.0f} MB, over Earth Engine's "
            f"{_MAX_RESPONSE_BYTES / 1024 ** 2:.0f} MB response limit. "
            "Reduce radius or coarsen scale."
        )

    west = float(x) - span
    north = float(y) + span
    grid = {
        "dimensions": {"width": width, "height": height},
        "affineTransform": {
            "scaleX": float(scale),
            "shearX": 0.0,
            "translateX": west,
            "shearY": 0.0,
            "scaleY": -float(scale),
            "translateY": north,
        },
        **spec,
    }

    xs = west + (np.arange(width, dtype=float) + 0.5) * scale
    ys = north - (np.arange(height, dtype=float) + 0.5) * scale
    return grid, xs, ys


def _compute_pixels(
    ee: ModuleType,
    image: Any,
    grid: dict[str, Any],
    origin: str,
) -> np.ndarray:
    """
    Pull one image onto `grid` as a structured numpy array.

    Parameters
    ----------
    ee : types.ModuleType
        The initialised Earth Engine module.
    image : ee.Image
        Image to evaluate; its bands become the fields of the result.
    grid : dict
        ``PixelGrid`` from :func:`_pixel_grid`.
    origin : str
        How to refer to the request in an error message.

    Returns
    -------
    numpy.ndarray
        Structured array of shape ``(height, width)``, one field per band.

    Raises
    ------
    RuntimeError
        If Earth Engine refuses or fails the request, with `origin` and the
        usual causes in the message.
    """
    try:
        return ee.data.computePixels(
            {
                "expression": image,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": grid,
            }
        )
    except Exception as exc:
        raise RuntimeError(
            f"Earth Engine could not return pixels for {origin}. Common causes "
            "are a request over the response size limit (reduce radius or "
            "coarsen scale), a quota or permission problem on the Cloud "
            f"project, or a transient service error. Earth Engine said: {exc}"
        ) from exc


def _as_dataarray(
    values: np.ndarray,
    valid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    parsed: CRS,
    name: str,
    attrs: dict[str, Any],
) -> xr.DataArray:
    """
    Wrap a fetched band and its mask as a georeferenced DataArray.

    Parameters
    ----------
    values : numpy.ndarray
        Band values, shape ``(height, width)``.
    valid : numpy.ndarray
        Mask on the same shape, non-zero where `values` is a real observation.
    xs, ys : numpy.ndarray
        Cell-centre coordinates from :func:`_pixel_grid`.
    parsed : pyproj.CRS
        CRS those coordinates are in.
    name : str
        Name for the array, which ``_align_raster`` carries through onto its
        own result.
    attrs : dict
        Provenance to record on the array.

    Returns
    -------
    xarray.DataArray
        Dims ``("y", "x")``, float64, ``nan`` wherever `valid` is zero, with
        the CRS and a ``nan`` nodata written through the ``.rio`` accessor.

    Raises
    ------
    ImportError
        If ``rioxarray`` is not installed, since without it the array cannot
        carry the CRS that ``_align_raster`` insists on.

    Notes
    -----
    Earth Engine returns masked pixels as zero, which for a class code or a
    vegetation index is indistinguishable from a real value; the companion mask
    band is what makes the gap recoverable, and turning it into ``nan`` here is
    what lets ``_align_raster`` hand back a truthful ``valid`` array.
    """
    _require("rioxarray")  # registers the .rio accessor used below

    data = np.where(np.asarray(valid) > 0, np.asarray(values, dtype="float64"), np.nan)
    array = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": ys, "x": xs},
        name=name,
        attrs=attrs,
    )
    array = array.rio.write_crs(parsed)
    array.rio.write_nodata(np.nan, inplace=True)
    return array


# ------------------------------
# Land cover
# ------------------------------


def fetch_nlcd(
    x: float,
    y: float,
    crs: str | int | CRS,
    year: int,
    radius: float = 3000.0,
    scale: float = DEFAULT_SCALE,
    collection: str = NLCD_COLLECTION,
    band: str = NLCD_BAND,
    project: str | None = None,
) -> xr.DataArray:
    """
    Fetch an NLCD land-cover tile centred on a tower.

    Reads one epoch of the National Land Cover Database from Earth Engine onto
    a square of half-width `radius` about ``(x, y)``, the input of the
    categorical evaluation of Chu et al. (2021), Sect. 2.4.

    Parameters
    ----------
    x, y : float
        Tower position in `crs` [m], e.g. the ``x_origin`` and ``y_origin`` of
        a :class:`~fluxfootprints.GridGeometry`.
    crs : str, int, or pyproj.CRS
        Projected CRS the coordinates are in, and the tile is returned in.
        Geographic CRSs are refused: `radius` and `scale` are metres.
    year : int
        NLCD release year to read, e.g. 2016 as in the paper. Must be an epoch
        `collection` publishes; the error message lists them.
    radius : float, default 3000.0
        Half-width of the square fetched [m]. The default circumscribes the
        largest target area of :data:`~fluxfootprints.TARGET_RADII`.
    scale : float, default 30.0
        Cell size [m]; 30 m is NLCD's native resolution.
    collection : str, default NLCD_COLLECTION
        Earth Engine ImageCollection to read the epoch from, should a newer
        release be wanted.
    band : str, default "landcover"
        Band of `collection` holding the class codes.
    project : str, optional
        Cloud project, forwarded to :func:`initialize` on first use.

    Returns
    -------
    xarray.DataArray
        Class codes on dims ``("y", "x")`` in `crs`, float64 with ``nan`` where
        NLCD is unclassified or absent, named ``"nlcd"`` and georeferenced for
        ``.rio``. Ready for ``_align_raster(..., categorical=True)``, which is
        what :func:`~fluxfootprints.sample_raster_on_grid` and
        :func:`~fluxfootprints.evaluate_landcover` build on.

    Raises
    ------
    ImportError
        If ``earthengine-api`` or ``rioxarray`` is not installed.
    ValueError
        If `crs` is geographic or unparseable, if `radius` or `scale` is not
        positive, if the grid would exceed the Earth Engine response limit, or
        if `year` is not an epoch of `collection`.
    RuntimeError
        If Earth Engine cannot be initialised, or refuses the request.

    See Also
    --------
    fetch_landsat_evi : The continuous field of the same analysis.
    fluxfootprints.evaluate_landcover : Consumes this, once warped onto the
        footprint grid.
    fluxfootprints.sample_raster_on_grid : Does that warp for a raster on disc.

    Notes
    -----
    Class codes must be resampled with nearest neighbour, so pass
    ``categorical=True`` when aligning: bilinear resampling of Anderson codes
    invents classes that are not in the legend. The codes themselves survive
    the float64 representation exactly.

    Chu et al. (2021) used NLCD 2016 over the conterminous United States, and
    the Canadian Land Cover map for their Canadian sites; only the former is
    fetched here, so a site outside NLCD's extent comes back all-``nan`` rather
    than wrong.

    Examples
    --------
    >>> geom = footprint_grid_geometry(model.x, model.y, 40.0, -111.9)   # doctest: +SKIP
    >>> nlcd = fetch_nlcd(geom.x_origin, geom.y_origin, geom.crs, 2016)  # doctest: +SKIP
    >>> aligned, valid = _align_raster(nlcd, grid, categorical=True)     # doctest: +SKIP

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4.
    """
    ee = initialize(project=project)
    parsed, spec = _crs_spec(crs)
    grid, xs, ys = _pixel_grid(x, y, radius, scale, spec, bands=2)

    epoch = str(int(year))
    source = ee.ImageCollection(collection)
    matched = source.filter(ee.Filter.eq("system:index", epoch))
    if int(matched.size().getInfo()) == 0:
        available = sorted(source.aggregate_array("system:index").getInfo())
        raise ValueError(
            f"{collection} publishes no epoch {epoch}; available epochs are "
            f"{', '.join(available) or '(none)'}."
        )

    landcover = ee.Image(matched.first()).select(band)
    # NLCD masks unclassified ground rather than coding it, and Earth Engine
    # returns a masked pixel as zero, so the mask travels alongside as the only
    # record of which zeros in the response are real class codes.
    request = landcover.rename("nlcd").addBands(landcover.mask().rename("valid"))
    fetched = _compute_pixels(ee, request, grid, f"NLCD {epoch} at ({x}, {y})")

    return _as_dataarray(
        fetched["nlcd"],
        fetched["valid"],
        xs,
        ys,
        parsed,
        name="nlcd",
        attrs={
            "long_name": "NLCD land cover class",
            "source": f"{collection}/{epoch}",
            "band": band,
            "year": int(year),
            "scale_m": float(scale),
            "radius_m": float(radius),
            "categorical": 1,
        },
    )


# ------------------------------
# Landsat EVI
# ------------------------------


def _timestamp(value: str | dt.date | dt.datetime | pd.Timestamp, label: str) -> str:
    """
    Normalise a date to the ``YYYY-MM-DD`` string Earth Engine filters on.

    Parameters
    ----------
    value : str, datetime.date, datetime.datetime, or pandas.Timestamp
        Date to normalise.
    label : str
        Argument name, for the error message.

    Returns
    -------
    str
        The date as ``YYYY-MM-DD``.

    Raises
    ------
    ValueError
        If `value` is not a date pandas can parse.
    """
    try:
        stamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(
            f"{label}={value!r} is not a date; pass 'YYYY-MM-DD', a "
            "datetime.date, or a pandas.Timestamp."
        ) from exc
    if pd.isna(stamp):
        raise ValueError(f"{label} must be a real date, got {value!r}.")
    return stamp.strftime("%Y-%m-%d")


def _bit_is_set(qa: Any, bit: int) -> Any:
    """
    Test one bit of a QA band.

    Parameters
    ----------
    qa : ee.Image
        Single-band integer image, ``QA_PIXEL``.
    bit : int
        Bit index, counting from zero as the Landsat Collection 2 product guide
        numbers them.

    Returns
    -------
    ee.Image
        Binary image, 1 where the bit is set.
    """
    return qa.rightShift(bit).bitwiseAnd(1)


def _obscured(qa: Any) -> Any:
    """
    Flag the pixels carrying no clear surface view.

    Parameters
    ----------
    qa : ee.Image
        The scene's ``QA_PIXEL`` band.

    Returns
    -------
    ee.Image
        Binary image named ``"obscured"``, 1 where the CFMask marks cloud,
        dilated cloud, cirrus, cloud shadow, or designated fill -- and 1
        outside the scene as well, since the unmasking is what makes a mean
        over the target disc count the part of the disc the swath never
        reached.

    Notes
    -----
    Folding fill and off-swath into the same flag as cloud is deliberate: a
    scene clipped by the swath edge is no more usable for a footprint-weighted
    mean than a clouded one, and screening both under a single threshold is
    what keeps a retrieved series comparable to the ``theme.coverage``
    filtering behind the paper's own data release.
    """
    flags = [_bit_is_set(qa, bit) for bit in (_QA_FILL_BIT, *_QA_OBSCURED_BITS)]
    combined = flags[0]
    for flag in flags[1:]:
        combined = combined.Or(flag)
    return combined.unmask(1).rename("obscured")


def _evi(image: Any, bands: LandsatBands) -> Any:
    """
    Compute EVI from one Collection 2 Level-2 scene, per Chu et al. Eq. 4.

    Parameters
    ----------
    image : ee.Image
        Scene from ``bands.collection``, with its reflectance bands still in
        their stored integer scaling.
    bands : LandsatBands
        Band names of the sensor that took the scene.

    Returns
    -------
    ee.Image
        Two-band image: ``evi``, masked to the clear pixels, and ``valid``, the
        mask itself, so a caller can tell a masked zero from a real zero.

    Notes
    -----
    ``2.5 (NIR - RED) / (NIR + 6 RED - 7.5 BLUE + 1)`` is defined on
    *reflectance*, not on stored digital numbers, so :data:`SR_SCALE` and
    :data:`SR_OFFSET` are applied first; skipping them would leave the ``+ 1``
    in the denominator negligible against the DNs and the index meaningless.

    The denominator is left as the equation states it. It can approach zero
    over bright bare surfaces, and the resulting outliers are the caller's to
    screen; clipping them here would quietly change the index the paper
    defines.
    """
    reflectance = (
        image.select([bands.blue, bands.red, bands.nir])
        .multiply(SR_SCALE)
        .add(SR_OFFSET)
    )
    clear = _obscured(image.select(bands.qa)).Not()

    evi = reflectance.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {
            "NIR": reflectance.select(bands.nir),
            "RED": reflectance.select(bands.red),
            "BLUE": reflectance.select(bands.blue),
        },
    ).rename("evi")

    masked = evi.updateMask(clear)
    return masked.addBands(masked.mask().rename("valid"))


def _clear_scenes(
    ee: ModuleType,
    bands: LandsatBands,
    disc: Any,
    first: str,
    last: str,
    scale: float,
    max_cloud: float,
    projection: str,
) -> list[tuple[pd.Timestamp, LandsatBands, str, float]]:
    """
    List one sensor's scenes that see the target disc essentially clear.

    Parameters
    ----------
    ee : types.ModuleType
        The initialised Earth Engine module.
    bands : LandsatBands
        Sensor to search.
    disc : ee.Geometry
        Target area the screening is applied over.
    first, last : str
        ``YYYY-MM-DD`` bounds, `first` inclusive and `last` exclusive.
    scale : float
        Resolution the obscured fraction is reduced at [m].
    max_cloud : float
        Fraction of `disc` allowed to lack a clear observation.
    projection : str
        CRS the reduction is carried out in, from :func:`_projection`.

    Returns
    -------
    list of tuple
        ``(acquired, bands, scene_id, obscured_fraction)`` per surviving scene,
        unordered; the caller sorts the merged sensors together.

    Notes
    -----
    The screening happens entirely server-side and comes back in a single
    ``getInfo``: only the scene ids and their obscured fractions cross the
    wire, and the pixels of the scenes that survive are fetched afterwards, one
    request each.
    """

    def tag(image: Any) -> Any:
        fraction = (
            _obscured(image.select(bands.qa))
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=disc,
                # crs and scale are pinned rather than inherited: unmasking
                # against a constant can leave the image's default projection
                # as degrees, and the fraction would then be reduced at a
                # resolution nothing in this module chose.
                crs=projection,
                scale=float(scale),
                bestEffort=False,
                maxPixels=int(1e9),
            )
            .get("obscured")
        )
        return image.set("obscured_fraction", fraction)

    kept = (
        ee.ImageCollection(bands.collection)
        .filterDate(first, last)
        .filterBounds(disc)
        .map(tag)
        .filter(ee.Filter.lt("obscured_fraction", float(max_cloud)))
    )

    metadata = ee.FeatureCollection(
        kept.map(
            lambda image: ee.Feature(
                None,
                {
                    "index": image.get("system:index"),
                    "time_start": image.get("system:time_start"),
                    "obscured_fraction": image.get("obscured_fraction"),
                },
            )
        )
    ).getInfo()

    scenes: list[tuple[pd.Timestamp, LandsatBands, str, float]] = []
    for feature in (metadata or {}).get("features", []):
        properties = feature.get("properties", {})
        fraction = properties.get("obscured_fraction")
        scenes.append(
            (
                pd.Timestamp(int(properties["time_start"]), unit="ms"),
                bands,
                str(properties["index"]),
                float("nan") if fraction is None else float(fraction),
            )
        )
    return scenes


def fetch_landsat_evi(
    x: float,
    y: float,
    crs: str | int | CRS,
    start: str | dt.date | dt.datetime | pd.Timestamp,
    end: str | dt.date | dt.datetime | pd.Timestamp,
    radius: float = 3000.0,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    scale: float = DEFAULT_SCALE,
    max_scenes: int | None = DEFAULT_MAX_SCENES,
    project: str | None = None,
) -> list[xr.DataArray]:
    """
    Fetch the clear Landsat EVI scenes over a tower, one array per scene.

    Merges Landsat 5 / TM and Landsat 8 / OLI Collection 2 Level-2 surface
    reflectance over ``[start, end)``, computes EVI per Chu et al. (2021),
    Eq. 4, keeps only the scenes that see the target disc essentially clear,
    and returns each on a tower-centred grid -- the input series of the
    continuous evaluation of Sect. 2.4.

    Parameters
    ----------
    x, y : float
        Tower position in `crs` [m].
    crs : str, int, or pyproj.CRS
        Projected CRS the coordinates are in, and the scenes are returned in.
        Geographic CRSs are refused.
    start, end : str, datetime.date, datetime.datetime, or pandas.Timestamp
        Retrieval window. `start` is inclusive and `end` exclusive, as
        ``ee.ImageCollection.filterDate`` reads them.
    radius : float, default 3000.0
        Half-width of the square fetched [m], and the radius of the disc the
        cloud screening is applied over.
    max_cloud : float, default 0.01
        Fraction of that disc allowed to lack a clear observation, i.e. to be
        cloud, cloud shadow, cirrus, fill, or off-swath. The default is the
        paper's < 1 % criterion.
    scale : float, default 30.0
        Cell size [m]; 30 m is Landsat's native resolution.
    max_scenes : int or None, default 200
        Refuse rather than issue more than this many pixel requests, since each
        surviving scene costs one. None lifts the cap.
    project : str, optional
        Cloud project, forwarded to :func:`initialize` on first use.

    Returns
    -------
    list of xarray.DataArray
        One array per surviving scene, oldest first, each on dims
        ``("y", "x")`` in `crs`, float64 with ``nan`` where the pixel was
        cloudy or off-swath, named ``"evi"`` and georeferenced for ``.rio``.
        Each carries a scalar ``time`` coordinate, and its scene id, sensor,
        and obscured fraction in ``.attrs``. Empty if no scene clears
        `max_cloud`.

    Raises
    ------
    ImportError
        If ``earthengine-api`` or ``rioxarray`` is not installed.
    ValueError
        If `crs` is geographic or unparseable; if `radius`, `scale`, or the
        dates are invalid; if `max_cloud` is outside ``[0, 1]``; if the grid
        would exceed the Earth Engine response limit; or if more than
        `max_scenes` scenes survive the screening.
    RuntimeError
        If Earth Engine cannot be initialised, or refuses a request.

    See Also
    --------
    fetch_nlcd : The categorical field of the same analysis.
    fluxfootprints.evaluate_vegetation_index : Consumes these, once warped onto
        the footprint grid and paired with matching climatologies.
    fluxfootprints.sensor_location_bias : Eq. 6, computed per matched scene.

    Notes
    -----
    Landsat 7 / ETM+ is skipped, as Chu et al. (2021) skip it: the scan-line
    corrector failed in 2003, and the resulting gaps would bias a
    footprint-weighted mean by however much of the footprint they fell on.
    Landsat 9 is likewise absent, since the paper's record ends in 2019 and
    this function reproduces its sensor set; a window after 2013 therefore
    returns Landsat 8 scenes alone.

    Pairing scenes to climatologies is the caller's job, as it is for
    :func:`~fluxfootprints.evaluate_vegetation_index`. The paper pairs each
    scene with the monthly climatology of the month it was retrieved in, which
    the ``time`` coordinate on each array supports directly.

    Each surviving scene is one ``computePixels`` request, so a wide window
    over a large radius is a slow call; narrowing the window is far cheaper
    than raising `max_scenes`.

    Examples
    --------
    >>> geom = footprint_grid_geometry(model.x, model.y, 40.0, -111.9)  # doctest: +SKIP
    >>> scenes = fetch_landsat_evi(                                    # doctest: +SKIP
    ...     geom.x_origin, geom.y_origin, geom.crs, "2015-01-01", "2020-01-01"
    ... )
    >>> [scene.attrs["date"] for scene in scenes[:2]]                  # doctest: +SKIP
    ['2015-04-12', '2015-06-15']

    References
    ----------
    Chu, H., et al. (2021). Agric. For. Meteorol., 301-302, 108350, Sect. 2.4,
    Eq. 4.
    """
    if not 0.0 <= max_cloud <= 1.0:
        raise ValueError(
            f"max_cloud is a fraction of the target disc and must lie in "
            f"[0, 1], got {max_cloud!r}."
        )

    first = _timestamp(start, "start")
    last = _timestamp(end, "end")
    if first > last:
        raise ValueError(f"start ({first}) must not fall after end ({last}).")

    ee = initialize(project=project)
    parsed, spec = _crs_spec(crs)
    grid, xs, ys = _pixel_grid(x, y, radius, scale, spec, bands=2)

    disc = ee.Geometry.Point([float(x), float(y)], proj=_projection(spec)).buffer(
        float(radius)
    )

    scenes: list[tuple[pd.Timestamp, LandsatBands, str, float]] = []
    for bands in LANDSAT_COLLECTIONS:
        scenes.extend(
            _clear_scenes(
                ee, bands, disc, first, last, scale, max_cloud, _projection(spec)
            )
        )

    if max_scenes is not None and len(scenes) > max_scenes:
        raise ValueError(
            f"{len(scenes)} scenes clear max_cloud={max_cloud} between {first} "
            f"and {last}, over the max_scenes={max_scenes} cap; each one is a "
            "separate pixel request. Narrow the window, or raise max_scenes."
        )

    # Sorted before fetching, so the expensive part runs in the order the
    # results are returned in and a part-finished call is still chronological.
    scenes.sort(key=lambda scene: (scene[0], scene[2]))

    fetched: list[xr.DataArray] = []
    for acquired, bands, scene_id, fraction in scenes:
        image = ee.Image(f"{bands.collection}/{scene_id}")
        values = _compute_pixels(
            ee, _evi(image, bands), grid, f"{bands.label} scene {scene_id}"
        )
        array = _as_dataarray(
            values["evi"],
            values["valid"],
            xs,
            ys,
            parsed,
            name="evi",
            attrs={
                "long_name": "Enhanced Vegetation Index",
                "source": f"{bands.collection}/{scene_id}",
                "spacecraft": bands.label,
                "scene_id": scene_id,
                "date": acquired.strftime("%Y-%m-%d"),
                "obscured_fraction": fraction,
                "scale_m": float(scale),
                "radius_m": float(radius),
                "equation": "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
            },
        )
        fetched.append(array.assign_coords(time=acquired))

    return fetched
