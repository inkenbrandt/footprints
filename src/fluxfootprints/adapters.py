# adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict
import numpy as np
import xarray as xr


import numpy as np
import xarray as xr
from .kljun import FFPModel as FootprintModel  # unified class


# --- COMMON OUTPUT CONTRACT ---
# Every adapter returns: Dict[int, Tuple[np.ndarray, np.ndarray]]
# where the dict key is the requested percent (e.g., 90) and
# the value is (xr, yr) arrays in **meters relative to tower**.
_REQUIRED = {
    "zm": "m",
    "h": "m",
    "ol": "m",
    "u_star": "m s-1",
    "sigma_v": "m s-1",
    # one of:
    # "z0" (m)  OR  "u_mean" (m s-1)
    # optional:
    "wind_dir": "degree",
}



@dataclass
class KM2001Params:
    zm: float
    z0: float
    L: float
    ustar: float
    umean: float
    sigma_v: float


def _iso_contour_from_2d(X, Y, F, fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return a single isopleth (x,y) enclosing `fraction` of mass."""
    # Flatten and find threshold with a mass–preserving cut
    dx = np.gradient(X[0, :])
    dy = np.gradient(Y[:, 0])
    cell = dx[None, :] * dy[:, None]
    flat = (F * cell).ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    thr = F.ravel()[order][np.searchsorted(csum / csum[-1], fraction)]
    # Extract contour (largest path) at this level
    import matplotlib.pyplot as plt

    cs = plt.contour(X, Y, F, levels=[thr])
    try:
        p = max(cs.collections[0].get_paths(), key=lambda p: p.vertices.shape[0])
        poly = p.vertices
        return poly[:, 0], poly[:, 1]
    finally:
        plt.close()


# ----------------- Kormann–Meixner (2001) -----------------
def km2001_isopleths(
    params: KM2001Params, rs: Iterable[float]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    import numpy as np
    from .kormannmeixner import (
        analytical_power_law_parameters,
        length_scale_xi,
        footprint_2d,
    )  #

    m, n, U, kappa = analytical_power_law_parameters(
        params.zm, params.z0, params.L, params.ustar, params.umean
    )  #
    xi = length_scale_xi(params.zm, U, kappa, m, n)  #

    # Build a compact grid (enough for fast, smooth contours)
    x = np.linspace(0.2, 3000.0, 400)
    y = np.linspace(-1000.0, 1000.0, 301)
    X, Y, F = footprint_2d(x, y, xi, m, n, params.umean, params.sigma_v)  #

    out = {}
    for r in rs:
        xr, yr = _iso_contour_from_2d(X, Y, F, r / 100.0)
        out[int(r)] = (xr, yr)
    return out


# ----------------- Wang et al. (2006) -----------------
@dataclass
class WangParams:
    zm: float
    h: float
    L: float
    sigma_v: float | None = None
    U: float | None = None


def wang2006_isopleths(
    params: WangParams, rs: Iterable[float]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    import numpy as np
    from .wang_footprint import wang2006_fy, reconstruct_gaussian_2d  #

    x = np.linspace(0.5, 3000.0, 400)
    fy = wang2006_fy(x, params.zm, params.h, params.L)  #
    X, Y, F = reconstruct_gaussian_2d(
        x, fy, sigma_v=params.sigma_v, U=params.U, ny=241
    )  #

    out = {}
    for r in rs:
        xr, yr = _iso_contour_from_2d(X, Y, F, r / 100.0)
        out[int(r)] = (xr, yr)
    return out


# ----------------- Kljun-style (your model.py) -----------------
def kljun2015_isopleths(
    row: dict, rs: Iterable[float]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Fast single-snapshot adapter using your model's equations to build one 2-D footprint
    and then cut mass-preserving isopleths, mirroring get_source_area_contour().
    """
    import numpy as np
    import xarray as xr
    from .kljun import FootprintModel  #

    # Build a tiny one-row DataFrame for the time step
    import pandas as pd

    df = pd.DataFrame(
        {
            "sigmav": [row["sigma_v"]],
            "ustar": [row["u_star"]],
            "ol": [row["L"]],
            "wind_dir": [row["wind_dir"]],
            "umean": [row["u_mean"]],
        }
    )
    fm = FootprintModel(
        df,
        domain=[-1200, 1200, -1200, 1200],
        dx=10,
        dy=10,
        smooth_data=False,
        verbosity=0,
    )
    f2d = fm.calc_xr_footprint()  # returns 2-D field on fm.x/fm.y (meters)

    X, Y = np.meshgrid(fm.x, fm.y, indexing="xy")
    out = {}
    for r in rs:
        xr, yr = _iso_contour_from_2d(
            X, Y, f2d.T.values, r / 100.0
        )  # transpose to (ny,nx)
        out[int(r)] = (xr, yr)
    return out



def _infer_meaning(ds: xr.Dataset) -> dict:
    # map common synonyms → canonical names; fail fast if missing
    rename = {}
    for k in list(ds.data_vars) + list(ds.coords):
        lk = k.lower()
        if lk in ("ustar", "u*"):
            rename[k] = "u_star"
        if lk in ("v_sigma", "sigmav", "sigma_v"):
            rename[k] = "sigma_v"
        if lk in ("mo_length", "l"):
            rename[k] = "ol"
        if lk in ("ws", "u_mean", "umean"):
            rename[k] = "u_mean"
        if lk in ("z0", "z0_roughness"):
            rename[k] = "z0"
        if lk in ("wd", "wind_dir", "winddirection"):
            rename[k] = "wind_dir"
    if rename:
        ds = ds.rename(rename)
    return ds


def ffp_xr(
    ds: xr.Dataset,
    *,
    rs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    nx: int = 1000,
    crop: bool = False,
    return_grid: bool = True,
    return_polygons: bool = False,
):
    """
    Run the unified footprint model over an xarray Dataset.

    Required variables (as data_vars or coords, per timestep):
    - zm, h, ol, u_star, sigma_v, and either z0 or u_mean.
    Optional: wind_dir.

    Returns
    -------
    If return_grid: DataArray with dims (time, y, x) of footprint weights.
    If return_polygons: DataArray with dim (time,) of GeoJSON strings for each r in rs.
    """
    ds = _infer_meaning(ds)

    # validate required fields
    has_z0 = "z0" in ds
    has_umean = "u_mean" in ds
    if not (has_z0 or has_umean):
        raise ValueError("Provide either 'z0' or 'u_mean' in the Dataset.")

    # Build a small runner that executes FootprintModel once
    def _run_one(df, zm, h, ol, sigma_v, u_star, z0, u_mean, wind_dir):
        # instantiate once per timestep; return a fixed grid ndarray
        model = FootprintModel(df
            zm=float(zm),
            h=float(h),
            ol=float(ol),
            sigmav=float(sigma_v),
            ustar=float(u_star),
            z0=float(z0) if np.isfinite(z0) else None,
            umean=float(u_mean) if np.isfinite(u_mean) else None,
            wind_dir=float(wind_dir) if np.isfinite(wind_dir) else None,
            nx=int(nx),
            rs=list(rs),
            crop=bool(crop),
        )
        out = model.grid  # (y, x) footprint weights
        return out

    # Prepare inputs (broadcast to time)
    req = ["zm", "h", "ol", "sigma_v", "u_star"]
    for k in req:
        if k not in ds:
            raise ValueError(f"Missing required variable: {k}")
    z0_arr = ds.get("z0", xr.full_like(ds["zm"], np.nan))
    um_arr = ds.get("u_mean", xr.full_like(ds["zm"], np.nan))
    wd_arr = ds.get("wind_dir", xr.zeros_like(ds["zm"]))

    grid = xr.apply_ufunc(
        _run_one,
        ds["zm"],
        ds["h"],
        ds["ol"],
        ds["sigma_v"],
        ds["u_star"],
        z0_arr,
        um_arr,
        wd_arr,
        input_core_dims=[[], [], [], [], [], [], [], []],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    grid.name = "footprint_weight"
    grid.attrs.update(
        units="m^-2", note="Kljun 2015 parameterisation via FootprintModel"
    )

    if not return_polygons:
        return grid

    # Optional: polygons per time (GeoJSON strings) generated by model
    # Keep this lightweight; do not recompute physics
    poly_json = xr.DataArray(
        [
            FootprintModel(
                zm=float(ds["zm"].isel(time=i)),
                h=float(ds["h"].isel(time=i)),
                ol=float(ds["ol"].isel(time=i)),
                sigmav=float(ds["sigma_v"].isel(time=i)),
                ustar=float(ds["u_star"].isel(time=i)),
                z0=(
                    float(z0_arr.isel(time=i))
                    if np.isfinite(z0_arr.isel(time=i))
                    else None
                ),
                umean=(
                    float(um_arr.isel(time=i))
                    if np.isfinite(um_arr.isel(time=i))
                    else None
                ),
                wind_dir=(
                    float(wd_arr.isel(time=i))
                    if np.isfinite(wd_arr.isel(time=i))
                    else None
                ),
                nx=int(nx),
                rs=list(rs),
                crop=bool(crop),
            ).geojson
            for i in range(ds.sizes["time"])
        ],
        dims=("time",),
        name="footprint_geojson",
    )
    return (grid if return_grid else None), poly_json
