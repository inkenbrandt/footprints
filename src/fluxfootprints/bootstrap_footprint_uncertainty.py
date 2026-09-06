"""
Block-bootstrap uncertainty for flux footprint climatologies.

Produces, on the same grid as the climatology itself:

  * ``p_include_{R}``  probability that a pixel falls inside the R% source area
  * ``w_mean``, ``w_p05``, ``w_p50``, ``w_p95``  per-pixel weight distribution
  * ``w_cv``           coefficient of variation (masked where the mean is tiny)

Design note -- why blocks are cached, not periods
-------------------------------------------------
A moving-block bootstrap resamples *blocks*, so every bootstrap member is a
non-negative integer combination of the per-block footprint sums.  We therefore
run the footprint model once per block and cache B rasters (B ~ 70 for a
season of daily blocks) instead of N rasters (N ~ 3300 half hours).  Members
are then formed by a single ``counts @ block_stack`` matmul, which makes
M = 500 members essentially free.

The block length must exceed the decorrelation time of the drivers that set
footprint geometry -- principally wind direction and the diurnal stability
cycle.  One to three days is the usual choice; an i.i.d. bootstrap over
half hours will badly understate the spread.

Normalisation
-------------
Per-period footprints are accumulated *raw* (not individually renormalised to
unity) and the member climatology is normalised once at the end.  Renormalising
each period first makes long, heavily truncated footprints contribute the same
total mass as short well-captured ones, which manufactures variance at the
domain edge.  The in-domain captured fraction is tracked per block and
propagated to each member so truncated members can be flagged or dropped.

Caveat: this module inherits whatever rotation convention the underlying model
uses.  Run it only after the ``arctan2`` argument-order fix -- a 90 deg
rotation is coherent across every member and the bootstrap will not reveal it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from scipy.ndimage import rotate as _rot

__all__ = [
    "BlockStack",
    "BootstrapResult",
    "GridSpec",
    "accumulate_blocks",
    "block_bootstrap",
    "rotate_climatology",
    "source_area_mask",
    "write_geotiff",
]


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    """Regular metric grid centred on the tower.

    ``x`` is positive east, ``y`` positive north, both in metres relative to
    the receptor.  Rasters are held in ``(row, col)`` order with row 0 at the
    *north* edge, i.e. GeoTIFF convention, not ``(x, y)``.
    """

    x: np.ndarray          # (nx,) cell centres, ascending east
    y: np.ndarray          # (ny,) cell centres, ascending north
    tower_lat: float
    tower_lon: float
    crs: str = "EPSG:32612"   # UTM 12N -- correct for Escalante, UT

    @classmethod
    def square(cls, half_width: float, dx: float, tower_lat: float,
               tower_lon: float, crs: str = "EPSG:32612") -> GridSpec:
        n = int(round(2 * half_width / dx))
        edge = (np.arange(n) - (n - 1) / 2) * dx
        return cls(x=edge, y=edge.copy(), tower_lat=tower_lat,
                   tower_lon=tower_lon, crs=crs)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    @property
    def dx(self) -> float:
        return float(np.diff(self.x)[0])

    @property
    def dy(self) -> float:
        return float(np.diff(self.y)[0])

    @property
    def cell_area(self) -> float:
        return abs(self.dx * self.dy)

    def to_raster_order(self, arr: np.ndarray, model_order: str = "xy") -> np.ndarray:
        """Convert model output to ``(row, col)`` with row 0 at the north edge. 

        Parameters
        ----------
        arr : numpy.ndarray
            2-D footprint array in either ``(x, y)`` or ``(y, x)`` order, depending on ``model_order``.
        model_order : str
            Order of the input array. Must be either 'xy' (x first, y second) or 'yx' (y first, x second). Default is 'xy'. 
        
        Returns
        -------
        numpy.ndarray
            2-D footprint array in ``(row, col)`` order with row 0 at the north edge, suitable for GeoTIFF output. The sum of the array is preserved.
        """
        a = np.asarray(arr)
        if model_order == "xy":
            a = a.T
        elif model_order != "yx":
            raise ValueError(f"model_order must be 'xy' or 'yx', got {model_order!r}")
        if a.shape != self.shape:
            raise ValueError(
                f"footprint shape {a.shape} does not match grid {self.shape}; "
                "check model_order and the model's domain settings"
            )
        return a[::-1, :]          # flip so row 0 is north


# --------------------------------------------------------------------------
# Phase 1 -- per-block accumulation
# --------------------------------------------------------------------------

@dataclass
class BlockStack:
    """Cached per-block footprint sums."""

    stack: np.ndarray            # (B, ny, nx) float32, raw summed weights
    n_periods: np.ndarray        # (B,) int, valid periods per block
    captured: np.ndarray         # (B,) float, summed in-domain mass
    labels: Sequence            # (B,) block identifiers, for traceability
    grid: GridSpec

    def __post_init__(self):
        keep = self.n_periods > 0
        if not keep.all():
            self.stack = self.stack[keep]
            self.n_periods = self.n_periods[keep]
            self.captured = self.captured[keep]
            self.labels = [l for l, k in zip(self.labels, keep) if k]

    @property
    def n_blocks(self) -> int:
        return self.stack.shape[0]

    @property
    def mean_capture(self) -> float:
        return float(self.captured.sum() / self.n_periods.sum())


def block_labels(index: pd.DatetimeIndex, block_days: int = 1) -> np.ndarray:
    """Contiguous calendar blocks of ``block_days`` length.
    
    Parameters
    ----------
    index : pandas.DatetimeIndex
        Input time index.
    block_days : int
        Length of blocks in days. Default is 1.

    Returns
    -------
    numpy.ndarray
        Integer block labels, same length as ``index``.  Each label is a contiguous block of ``block_days`` days, starting from the first day in ``index``.
    """
    origin = index.min().normalize()
    return ((index.normalize() - origin).days // block_days).to_numpy()


def accumulate_blocks(
    df: pd.DataFrame,
    grid: GridSpec,
    period_footprint: Callable[[pd.Series], np.ndarray | None],
    block_days: int = 1,
    model_order: str = "xy",
    progress: bool = True,
) -> BlockStack:
    """Accumulate per-block footprint sums for later bootstrap resampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Input meteorological data, indexed by a DatetimeIndex.
    grid : GridSpec
        The grid on which the climatology is defined.
    period_footprint : callable
        Function that takes a row of ``df`` and returns a 2-D footprint density on the grid (units m^-2), or None if the period is unusable.
    block_days : int
        Length of blocks in days. Default is 1.
    model_order : str
        Order of the footprint array returned by ``period_footprint``. Must be either 'xy' (default) or 'yx'.
    progress : bool
        If True, print progress messages during accumulation.

    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a DatetimeIndex")

    labels = block_labels(df.index, block_days)
    uniq = np.unique(labels)
    ny, nx = grid.shape

    stack = np.zeros((uniq.size, ny, nx), dtype=np.float32)
    n_per = np.zeros(uniq.size, dtype=np.int64)
    cap = np.zeros(uniq.size, dtype=np.float64)
    lookup = {b: i for i, b in enumerate(uniq)}

    n_fail = 0
    for step, (ts, row) in enumerate(df.iterrows()):
        try:
            fp = period_footprint(row)
        except Exception as exc:                      # noqa: BLE001
            warnings.warn(f"{ts}: footprint failed ({exc})", stacklevel=2)
            fp = None
        if fp is None:
            n_fail += 1
            continue
        fp = grid.to_raster_order(fp, model_order).astype(np.float32)
        if not np.isfinite(fp).all():
            n_fail += 1
            continue
        i = lookup[labels[step]]
        stack[i] += fp
        n_per[i] += 1
        cap[i] += float(fp.sum()) * grid.cell_area
        if progress and step % 250 == 0:
            print(f"  {step}/{len(df)} periods", flush=True)

    if progress:
        print(f"  done: {int(n_per.sum())} usable, {n_fail} skipped, "
              f"{int((n_per > 0).sum())} blocks", flush=True)

    return BlockStack(stack=stack, n_periods=n_per, captured=cap,
                      labels=list(uniq), grid=grid)


# --------------------------------------------------------------------------
# Source areas
# --------------------------------------------------------------------------

def source_area_mask(w: np.ndarray, R: float) -> np.ndarray:
    """Boolean mask of the smallest set of pixels holding fraction ``R``.

    ``w`` is a non-negative weight raster.  ``R`` is interpreted relative to
    the *in-domain* total, so check the captured fraction before trusting an
    R close to 1.

    Parameters
    ----------
    w : numpy.ndarray
        2-D footprint weights on the grid, in (row, col) order. 
    R : float
        Source-area fraction to enclose, in (0, 1).
    
    Returns
    -------
    numpy.ndarray
        Boolean mask of the same shape as ``w``.  True where the pixel is part of the smallest set of pixels that enclose at least ``R`` of the total
        in-domain source weight.

    """
    flat = w.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order], dtype=np.float64)
    total = csum[-1]
    if total <= 0:
        return np.zeros_like(w, dtype=bool)
    k = int(np.searchsorted(csum, R * total))
    k = min(k, flat.size - 1)
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[: k + 1]] = True
    return mask.reshape(w.shape)


# --------------------------------------------------------------------------
# Phase 2 -- bootstrap
# --------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    grid: GridSpec
    levels: tuple[float, ...]
    p_include: dict[float, np.ndarray]      # R -> (ny, nx) in [0, 1]
    w_mean: np.ndarray
    w_p05: np.ndarray
    w_p50: np.ndarray
    w_p95: np.ndarray
    w_cv: np.ndarray
    member_capture: np.ndarray              # (M,)
    n_members: int
    n_dropped: int
    meta: dict = field(default_factory=dict)

    """
    BootstrapResult holds the output of :func:`block_bootstrap`, which resamples
    cached per-block footprint sums to produce an ensemble of climatologies and
    summarises the per-pixel distribution.  The ``p_include_{R}`` bands
    give the probability that a pixel falls inside the R% source area, and the
    ``w_*`` bands summarise the per-pixel weight distribution.  The ``member_capture`` array gives the in-domain captured fraction for each member, which can be used to filter out
    poorly performing members.

    Parameters
    ----------
    grid : GridSpec
        The grid on which the climatology is defined.
    levels : tuple of float
        Source-area fractions to evaluate, in (0, 1).
    p_include : dict of float to numpy.ndarray
        Probability that a pixel falls inside the R% source area, for each R in levels.
    w_mean : numpy.ndarray
        Mean weight per pixel across the bootstrap members.
    w_p05 : numpy.ndarray
        5th percentile weight per pixel across the bootstrap members.
    w_p50 : numpy.ndarray
        50th percentile (median) weight per pixel across the bootstrap members.
    w_p95 : numpy.ndarray
        95th percentile weight per pixel across the bootstrap members.
    w_cv : numpy.ndarray
        Coefficient of variation (std / mean) per pixel across the bootstrap members, masked where the mean is below a specified quantile.
    member_capture : numpy.ndarray
        In-domain captured fraction for each bootstrap member.
    n_members : int
        Number of bootstrap members that were kept after filtering.
    n_dropped : int
        Number of bootstrap members that were dropped due to failing the capture test.
    meta : dict
        Additional metadata about the bootstrap process, including the number of blocks, total periods, mean capture, minimum capture threshold, coefficient of variation floor, and random seed used.
    """

    def bands(self) -> list[tuple[str, np.ndarray]]:
        out = [(f"p_include_{int(round(R * 100))}", self.p_include[R])
               for R in self.levels]
        out += [("w_mean", self.w_mean), ("w_p05", self.w_p05),
                ("w_p50", self.w_p50), ("w_p95", self.w_p95),
                ("w_cv", self.w_cv)]
        return out


def block_bootstrap(
    blocks: BlockStack,
    n_members: int = 500,
    levels: Sequence[float] = (0.5, 0.8, 0.9),
    min_capture: float | None = None,
    cv_floor_quantile: float = 0.50,
    seed: int | None = 0,
) -> BootstrapResult:
    """Resample blocks with replacement and summarise the member ensemble.

    Parameters
    ----------
    blocks : BlockStack
        Cached per-block footprint sums.
    n_members : int
        Number of bootstrap members to generate.
    levels : sequence of float
        Source-area fractions to evaluate, in (0, 1).
    min_capture : float or None
        Minimum in-domain capture fraction for a member to be kept.  If None,
        defaults to ``max(levels) + 0.05``.
    cv_floor_quantile : float
        Quantile of the mean weight below which the coefficient of variation is
        masked (NaN).  Default is 0.5 (median).
    seed : int or None
        Random seed for reproducibility.  If None, the RNG is not seeded.
    """
    rng = np.random.default_rng(seed)
    levels = tuple(sorted(float(R) for R in levels))
    if min_capture is None:
        min_capture = min(0.99, max(levels) + 0.05)

    B = blocks.n_blocks
    if B < 20:
        warnings.warn(
            f"only {B} blocks; block-bootstrap intervals are unreliable below "
            "~20-30 blocks. Shorten block_days or accept wider uncertainty.",
            stacklevel=2,
        )

    ny, nx = blocks.grid.shape
    npix = ny * nx
    area = blocks.grid.cell_area
    flat_stack = blocks.stack.reshape(B, npix)          # (B, npix) float32

    counts = rng.multinomial(B, np.full(B, 1.0 / B), size=n_members)  # (M, B)

    members = np.empty((n_members, npix), dtype=np.float32)
    capture = np.empty(n_members, dtype=np.float64)
    incl = {R: np.zeros(npix, dtype=np.int32) for R in levels}
    keep = np.ones(n_members, dtype=bool)

    chunk = max(1, int(2e8 // (npix * 4)))              # cap the matmul at ~200 MB
    for s in range(0, n_members, chunk):
        e = min(s + chunk, n_members)
        raw = counts[s:e].astype(np.float32) @ flat_stack       # (m, npix)
        tot = raw.sum(axis=1, dtype=np.float64) * area
        n_p = counts[s:e] @ blocks.n_periods
        cap = counts[s:e] @ blocks.captured
        capture[s:e] = np.where(n_p > 0, cap / np.maximum(n_p, 1), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            members[s:e] = raw / np.where(tot > 0, tot, np.nan)[:, None]

    for m in range(n_members):
        if not np.isfinite(members[m]).all() or capture[m] < min_capture:
            keep[m] = False
            continue
        w2d = members[m].reshape(ny, nx)
        for R in levels:
            incl[R] += source_area_mask(w2d, R).ravel()

    n_keep = int(keep.sum())
    if n_keep == 0:
        raise RuntimeError(
            "every bootstrap member failed the capture test; the domain is far "
            "too small for these measurement heights"
        )
    kept = members[keep]

    p_incl = {R: (incl[R] / n_keep).reshape(ny, nx).astype(np.float32)
              for R in levels}
    w_mean = kept.mean(axis=0)
    q05, q50, q95 = np.percentile(kept, [5, 50, 95], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = kept.std(axis=0, ddof=1) / w_mean
    floor = np.quantile(w_mean[w_mean > 0], cv_floor_quantile) if (w_mean > 0).any() else 0.0
    cv = np.where(w_mean >= floor, cv, np.nan)

    rs = lambda a: np.asarray(a, dtype=np.float32).reshape(ny, nx)
    return BootstrapResult(
        grid=blocks.grid,
        levels=levels,
        p_include=p_incl,
        w_mean=rs(w_mean), w_p05=rs(q05), w_p50=rs(q50), w_p95=rs(q95),
        w_cv=rs(cv),
        member_capture=capture[keep],
        n_members=n_keep,
        n_dropped=int(n_members - n_keep),
        meta={
            "n_blocks": B,
            "n_periods": int(blocks.n_periods.sum()),
            "mean_capture": blocks.mean_capture,
            "min_capture": min_capture,
            "cv_floor": float(floor),
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------
# Sonic azimuth bias - constant, not noise
# --------------------------------------------------------------------------

def rotate_climatology(w: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate the climatology about the tower by a constant azimuth offset.

    Parameters
    ----------
    w : np.ndarray
        2-D footprint climatology on the grid, in (row, col) order.
    degrees : float
        Azimuth rotation in degrees, positive clockwise; A sonic azimuth error rotates every half-hourly footprint by the same
    angle

    Returns
    -------
    np.ndarray
        Rotated footprint climatology, same shape as ``w``.  The sum is preserved, but the in-domain captured fraction may change.
    """
    

    out = _rot(w, angle=-degrees, reshape=False, order=1,
               mode="constant", cval=0.0)
    np.clip(out, 0.0, None, out=out)
    s = out.sum()
    return (out / s * w.sum()).astype(np.float32) if s > 0 else out.astype(np.float32)


# --------------------------------------------------------------------------
# Phase 3 -- GeoTIFF
# --------------------------------------------------------------------------

def write_geotiff(result: BootstrapResult, path: str, compress: str = "deflate") -> str:
    """Write all bands to one multiband float32 GeoTIFF.
    
    Parameters
    ----------
    result : BootstrapResult
        The result of :func:`block_bootstrap`.
    path : str
        Output file path.
    compress : str
        Compression method for the GeoTIFF. Default is "deflate". Other options include "lzw", "jpeg", "packbits", etc. See rasterio documentation for supported compression methods
    
    Returns
    -------
    str
        The path to the written GeoTIFF file.
    """


    g = result.grid
    tx = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True)
    e0, n0 = tx.transform(g.tower_lon, g.tower_lat)

    transform = (
        Affine.translation(e0 + g.x.min() - g.dx / 2.0,
                           n0 + g.y.max() + g.dy / 2.0)
        * Affine.scale(g.dx, -g.dy)
    )

    bands = result.bands()
    profile = {
        "driver": "GTiff", "height": g.shape[0], "width": g.shape[1],
        "count": len(bands), "dtype": "float32", "crs": g.crs, "transform": transform,
        "nodata": np.float32(np.nan), "compress": compress, "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }

    with rasterio.open(path, "w", **profile) as dst:
        for i, (name, arr) in enumerate(bands, start=1):
            dst.write(np.asarray(arr, dtype=np.float32), i)
            dst.set_band_description(i, name)
        dst.update_tags(
            **{k: str(v) for k, v in result.meta.items()},
            n_members=str(result.n_members),
            n_dropped=str(result.n_dropped),
            levels=",".join(str(R) for R in result.levels),
            note="weights are m^-2 densities; p_include is a probability in [0,1]",
        )
    return path


# --------------------------------------------------------------------------
# Adapter -- the one place to wire in fluxfootprints
# --------------------------------------------------------------------------

def ffp_period_adapter(grid: GridSpec, z_m: float, z_0: float,
                       model_cls=None) -> Callable[[pd.Series], np.ndarray | None]:
    """ Return a function that runs the footprint model for one period.

    The returned function takes a row of the input DataFrame and returns a 2-D footprint density on the grid (units m^-2), or None if the period is unusable.

    Parameters
    ----------
    grid : GridSpec
        The grid on which the climatology is defined.
    z_m : float
        Measurement height minus displacement height [m].
    z_0 : float
        Roughness length [m].
    model_cls : class, optional
        The footprint model class to use. If None, defaults to fluxfootprints.improved_ffp.FFPModel.
    
    Returns
    -------
    Callable[[pd.Series], np.ndarray | None]
        A function that takes a row of the input DataFrame and returns a 2-D footprint density on the grid (units m^-2), or None if the period is unusable.
    """
    if model_cls is None:
        from fluxfootprints.improved_ffp import FFPModel as model_cls

    req = ("WS", "WD", "USTAR", "MO_LENGTH", "V_SIGMA", "PBLH_F")

    def _fn(row: pd.Series) -> np.ndarray | None:
        if any(pd.isna(row.get(c)) for c in req):
            return None
        L, h, ustar = float(row.MO_LENGTH), float(row.PBLH_F), float(row.USTAR)
        if h <= 10.0 or ustar <= 0.1 or float(row.V_SIGMA) <= 0:
            return None
        if not (20.0 * z_0 < z_m < 0.8 * h):        # Kljun eq. 27
            return None
        if z_m / L < -15.5:
            return None

        model = model_cls(
            zm=z_m, z0=z_0, h=h, ol=L, ustar=ustar,
            sigmav=float(row.V_SIGMA), umean=float(row.WS),
            wind_dir=float(row.WD),
            domain=(grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()),
            dx=grid.dx, dy=grid.dy,
        )
        res = model.run()
        fp = res["fclim_2d"] if isinstance(res, dict) else res
        return None if fp is None else np.asarray(fp)

    return _fn


# --------------------------------------------------------------------------

if __name__ == "__main__":
    CSV = "/mnt/project/USUTE_HH_202406241430_202409251400.csv"
    Z_M, Z_0 = 2.0, 0.05          # receptor height minus d, and roughness

    df = pd.read_csv(CSV, na_values=[-9999.0])
    df.index = pd.to_datetime(df.TIMESTAMP_START.astype("int64").astype(str),
                              format="%Y%m%d%H%M")

    grid = GridSpec.square(half_width=600.0, dx=5.0,
                           tower_lat=37.7353, tower_lon=-111.5708)

    blocks = accumulate_blocks(
        df, grid, ffp_period_adapter(grid, Z_M, Z_0), block_days=1,
    )
    print(f"blocks={blocks.n_blocks}  mean capture={blocks.mean_capture:.3f}")

    res = block_bootstrap(blocks, n_members=500, levels=(0.5, 0.8, 0.9))
    print(f"members kept {res.n_members}, dropped {res.n_dropped}")

    write_geotiff(res, "/mnt/user-data/outputs/US-UTE_footprint_uncertainty.tif")

    for deg in (-10, -5, 5, 10):
        _ = rotate_climatology(res.w_mean, deg)   # azimuth scenarios
