"""footprint_compare.py
========================
Utility functions for computing and comparing eddy‑covariance flux‑footprint
models.

The module provides a small toolkit to **re‑run the side‑by‑side evaluations**
we discussed earlier:

1. Run multiple footprint models (classic Kljun‑FFP, the xarray rewrite
   ``ffp_xr``\, the analytic Kormann‑Meixner (KM) approximation, or any other
   user‑supplied function).
2. Compute common diagnostic metrics against a chosen reference model
   (RMSE, peak‑location bias, 80 % source‑area overlap).
3. Return a tidy ``pandas.DataFrame`` *and* create optional difference plots.

All heavy lifting (meteorological inputs, domain definition …) is delegated to
existing model APIs; this file is purely orchestration + diagnostics.

Example
-------
>>> import pandas as pd, numpy as np
>>> from footprint_compare import (
...     run_ffp, run_ffp_xr, run_km, compare_footprints)
>>> met = pd.DataFrame({  # 24 h synthetic
...     "wind_dir": 0.0, "ws": 5.0, "sigmav": .5,
...     "ustar": .3, "ol": 200.0},
...     index=pd.date_range("2025-05-01", periods=24, freq="h"))
>>> dom = [-300, 300, -300, 300]
>>> f_ref, (x, y) = run_ffp(met, domain=dom, dx=2)
>>> f_xr,  _       = run_ffp_xr(met, domain=dom, dx=2)
>>> res = compare_footprints(f_ref, (x, y),
...                          {"FFP_xr": (f_xr, (x, y))})
>>> print(res)

Dependencies
------------
``numpy, pandas, matplotlib, scipy``;
footprint models: ``calc_footprint_FFP_climatology``, ``ffp_xr`` (these must be
importable, e.g. placed on ``PYTHONPATH``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# -----------------------------------------------------------------------------
# Helper metrics
# -----------------------------------------------------------------------------


def _peak_distance(f: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Euclidean distance from the origin to the peak of a footprint.

    Identifies the location of the maximum value in the footprint array `f`
    and calculates its distance from the origin (assumed tower location at (0, 0))
    using the provided coordinate grids.

    Parameters
    ----------
    f : numpy.ndarray
        2D footprint array with source strength values.
    x : numpy.ndarray
        2D array of x-coordinate values corresponding to `f`.
    y : numpy.ndarray
        2D array of y-coordinate values corresponding to `f`.

    Returns
    -------
    distance : float
        Euclidean distance from the origin to the peak footprint location [m].
    """
    idx = np.unravel_index(np.argmax(f), f.shape)
    return float(np.hypot(x[idx], y[idx]))


def _mask_80(f: np.ndarray) -> np.ndarray:
    """
    Create a boolean mask for the 80% cumulative contribution area of a footprint.

    Identifies the cells in the footprint array `f` that together contribute
    to 80% of the total flux source area by cumulative sum of values sorted
    in descending order.

    Parameters
    ----------
    f : numpy.ndarray
        2D footprint array representing flux contributions.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of the same shape as `f`, where `True` indicates cells
        that belong to the 80% cumulative source area.
    """

    flat = f.flatten()
    idx_sort = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx_sort])
    thresh = flat[idx_sort][np.searchsorted(cumsum, 0.8 * flat.sum())]
    return f >= thresh


def footprint_metrics(
    f_ref: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    f_other: np.ndarray,
    x_other: np.ndarray,
    y_other: np.ndarray,
) -> Dict[str, float]:
    """
    Compute diagnostic comparison metrics between two footprint distributions.

    Compares a test footprint (`f_other`) against a reference footprint (`f_ref`)
    using several spatial metrics: root mean squared error (RMSE), peak location difference,
    and percentage overlap of the 80% contribution area.

    Parameters
    ----------
    f_ref : numpy.ndarray
        Reference footprint values on a 2D grid.
    x_ref : numpy.ndarray
        X-coordinates for the reference footprint grid.
    y_ref : numpy.ndarray
        Y-coordinates for the reference footprint grid.
    f_other : numpy.ndarray
        Footprint values from the model being compared.
    x_other : numpy.ndarray
        X-coordinates for the comparison footprint grid.
    y_other : numpy.ndarray
        Y-coordinates for the comparison footprint grid.

    Returns
    -------
    metrics : dict of str to float
        Dictionary containing:
        - 'RMSE' : Root mean squared error between footprints.
        - 'Peak_ref' : Distance of the peak location in the reference footprint from the origin [m].
        - 'Peak_other' : Distance of the peak location in the comparison footprint from the origin [m].
        - 'Peak_diff' : Absolute difference in peak distances [m].
        - 'Overlap80(%)' : Percent overlap of 80% contribution regions [%].

    Raises
    ------
    ValueError
        If the input grids are not congruent in shape.
    """
    if not (
        f_ref.shape
        == f_other.shape
        == x_ref.shape
        == y_ref.shape
        == x_other.shape
        == y_other.shape
    ):
        raise ValueError(
            "Grids must be congruent. Ensure models use identical x/y domain."
        )

    rmse = float(np.sqrt(np.mean((f_other - f_ref) ** 2)))
    peak_ref = _peak_distance(f_ref, x_ref, y_ref)
    peak_other = _peak_distance(f_other, x_other, y_other)
    peak_diff = abs(peak_other - peak_ref)

    mask_ref = _mask_80(f_ref)
    mask_other = _mask_80(f_other)
    overlap = float((mask_ref & mask_other).sum() / mask_ref.sum() * 100.0)

    return {
        "RMSE": rmse,
        "Peak_ref": peak_ref,
        "Peak_other": peak_other,
        "Peak_diff": peak_diff,
        "Overlap80(%)": overlap,
    }


# -----------------------------------------------------------------------------
# Wrappers to *run* individual models
# -----------------------------------------------------------------------------

# Classic FFP (Kljun 2015) -----------------------------------------------------


def run_ffp(
    met: pd.DataFrame,
    *,
    zm: float = 2.0,
    z0: float = 0.1,
    h: float = 2000.0,
    domain: list[float] | Tuple[float, float, float, float] = (-300, 300, -300, 300),
    dx: float = 2.0,
    **extra_kwargs,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the footprint climatology using the classic FFP implementation.

    This function runs the `FFP_climatology` model from the
    `calc_footprint_FFP_climatology` module, generating a 2D footprint
    and corresponding spatial grid arrays based on provided meteorological inputs.

    Parameters
    ----------
    met : pandas.DataFrame
        DataFrame containing time series of meteorological variables.
        Expected columns include:
        - 'ol' : Obukhov length [m]
        - 'sigmav' : Standard deviation of lateral velocity [m/s]
        - 'ustar' : Friction velocity [m/s]
        - 'wind_dir' : Wind direction [degrees]
    zm : float, optional
        Measurement height above displacement height [m]. Default is 2.0.
    z0 : float, optional
        Surface roughness length [m]. Default is 0.1.
    h : float, optional
        Boundary layer height [m]. Default is 2000.0.
    domain : list of float or tuple of float, optional
        Spatial extent of the footprint domain in the format
        (xmin, xmax, ymin, ymax). Default is (-300, 300, -300, 300).
    dx : float, optional
        Grid resolution in both x and y directions [m]. Default is 2.0.
    **extra_kwargs
        Additional keyword arguments passed to `FFP_climatology`.

    Returns
    -------
    ffp_2d : numpy.ndarray
        2D array of footprint climatology values.
    (x, y) : tuple of numpy.ndarray
        Tuple of 2D coordinate arrays for the x and y spatial grid [m].
    """

    from .volk import ffp_climatology as _FFP

    ts_len = len(met)
    out = _FFP(
        zm=[zm] * ts_len,
        z0=[z0] * ts_len,
        umean=None,
        h=[h] * ts_len,
        ol=met["ol"].tolist() if "ol" in met else [200.0] * ts_len,
        sigmav=met.get("sigmav", pd.Series(0.5, index=met.index)).tolist(),
        ustar=met.get("ustar", pd.Series(0.3, index=met.index)).tolist(),
        wind_dir=met.get("wind_dir", pd.Series(0.0, index=met.index)).tolist(),
        domain=list(domain),
        dx=dx,
        dy=dx,
        smooth_data=0,
        crop=False,
        verbosity=0,
        fig=False,
        **extra_kwargs,
    )
    return np.asarray(out["fclim_2d"]), (
        np.asarray(out["x_2d"]),
        np.asarray(out["y_2d"]),
    )


# xarray rewrite (ffp_xr.py) ---------------------------------------------------


def run_ffp_xr(
    met: pd.DataFrame,
    *,
    domain: list[float] | Tuple[float, float, float, float] = (-300, 300, -300, 300),
    dx: float = 2.0,
    logger=None,
    **extra_kwargs,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the footprint climatology using the xarray-based FFP implementation.

    This function runs the `ffp_climatology_new` model from the `ffp_xr` module
    on a meteorological input DataFrame, returning the 2D footprint
    and associated spatial grid coordinates.

    Parameters
    ----------
    met : pandas.DataFrame
        DataFrame containing meteorological inputs for footprint modeling.
    domain : list of float or tuple of float, optional
        Spatial extent of the footprint domain in the format
        (xmin, xmax, ymin, ymax). Default is (-300, 300, -300, 300).
    dx : float, optional
        Grid resolution in both x and y directions. Default is 2.0.
    logger : logging.Logger or None, optional
        Logger instance for model output. If None, a default logger is created.
    **extra_kwargs
        Additional keyword arguments passed to `ffp_climatology_new`.

    Returns
    -------
    ffp_2d : numpy.ndarray
        2D array representing the climatological footprint values.
    (x, y) : tuple of numpy.ndarray
        Tuple of arrays representing the x and y coordinates of the footprint grid.
    """

    from ffp_xr import ffp_climatology_new as _FFPXR

    if logger is None:
        import logging

        logger = logging.getLogger("ffp_xr")
        logger.setLevel(logging.ERROR)

    model = _FFPXR(
        df=met.copy(),
        domain=list(domain),
        dx=dx,
        dy=dx,
        smooth_data=False,
        verbosity=0,
        logger=logger,
        **extra_kwargs,
    )
    model.calc_xr_footprint()
    return model.fclim_2d.values, (model.xv, model.yv)


# Analytic KM (Kormann & Meixner) --------------------------------------------


def run_km(
    met: pd.DataFrame,
    *,
    zm: float = 2.0,
    z0: float = 0.1,
    domain: list[float] | Tuple[float, float, float, float] = (-300, 300, -300, 300),
    dx: float = 2.0,
    **extra_kwargs,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute a toy 2D footprint based on a Kormann–Meixner-style analytic solution.

    This simplified implementation models the footprint as a product of a
    1D Gamma distribution in the along-wind (y) direction and a Gaussian in the
    cross-wind (x) direction. It is intended as a minimal, conceptual reference.

    Parameters
    ----------
    met : pandas.DataFrame
        Meteorological input data (unused in this implementation, but included for API compatibility).
    zm : float, optional
        Measurement height above displacement height [m]. Default is 2.0.
    z0 : float, optional
        Surface roughness length [m]. Default is 0.1.
    domain : list of float or tuple of float, optional
        Spatial extent of the footprint domain in the format
        (xmin, xmax, ymin, ymax). Default is (-300, 300, -300, 300).
    dx : float, optional
        Grid resolution in both x and y directions [m]. Default is 2.0.
    **extra_kwargs
        Ignored in this implementation. Included for API consistency.

    Returns
    -------
    f2d : numpy.ndarray
        2D array of normalized footprint values.
    (xv, yv) : tuple of numpy.ndarray
        Tuple of 2D arrays representing the x and y coordinates of the grid [m].

    Notes
    -----
    - The analytic form is loosely inspired by Kormann & Meixner (2001) under neutral conditions.
    - Not intended for production use; serves as a placeholder or toy model.
    """
    # Build grid
    xmin, xmax, ymin, ymax = domain
    x = np.arange(xmin, xmax + dx, dx)
    y = np.arange(ymin, ymax + dx, dx)
    xv, yv = np.meshgrid(x, y)

    # Parameters (roughly tuned to mimic FFP peak)
    m, n = 1.0, 0.5  # power‑law exponents for u, K (neutral)
    r = 2 + m - n
    U = 1.0  # non‑dimensional wind speed
    kappa = 0.4
    xi = (U * zm**r) / (r**2 * kappa * z0**n)

    # Cross‑wind Gaussian std ~  σ = a * |x|
    a = 0.35

    fy = (xi ** ((1 + m) / r)) / (r * np.math.gamma((1 + m) / r))
    fy = fy * (np.abs(yv) ** (m / r - 1)) * np.exp(-xi / np.abs(yv))

    cross = (1 / (np.sqrt(2 * np.pi) * a * np.abs(yv))) * np.exp(
        -(xv**2) / (2 * (a**2) * yv**2)
    )

    f2d = fy * cross
    f2d[np.isnan(f2d)] = 0.0
    f2d /= f2d.sum() * dx**2  # normalize
    return f2d, (xv, yv)


# -----------------------------------------------------------------------------
# High‑level comparison utilities
# -----------------------------------------------------------------------------


def compare_footprints(
    ref_f: np.ndarray,
    ref_grid: Tuple[np.ndarray, np.ndarray],
    others: Dict[str, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    *,
    plot: bool = True,
    cmap="RdBu",
    vlim: float | None = None,
) -> pd.DataFrame:
    """Compute metrics for *others* against reference; optionally plot diffs.

    Parameters
    ----------
    ref_f / ref_grid
        Reference footprint density and its (x_2d, y_2d) grids.
    others
        Dict mapping *name* → (f_2d, (x_2d, y_2d)).
    plot
        If True, draw a difference map for each entry.
    vlim
        Symmetric colour‑bar limit for difference plots.  By default the max
        absolute difference across all models is used.
    Returns
    -------
    ``pandas.DataFrame`` with rows = model names, columns = metrics.
    """
    x_ref, y_ref = ref_grid
    records = {}

    if plot:
        nrow = len(others)
        fig, axes = plt.subplots(1, nrow, figsize=(5 * nrow, 4), squeeze=False)

    for i, (name, (f, (x, y))) in enumerate(others.items()):
        rec = footprint_metrics(ref_f, x_ref, y_ref, f, x, y)
        records[name] = rec

        if plot:
            ax = axes[0, i]
            diff = f - ref_f
            vmax = vlim or np.max(np.abs(diff))
            pcm = ax.pcolormesh(x, y, diff, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_title(f"{name} – ref")
            if i == nrow - 1:
                plt.colorbar(pcm, ax=ax, label="Δ density")

    if plot:
        plt.tight_layout()

    df = pd.DataFrame.from_dict(records, orient="index")
    return df


# -----------------------------------------------------------------------------
# Convenience: run *all* models on same met‑data & compare in one call
# -----------------------------------------------------------------------------


def run_all_and_compare(
    met: pd.DataFrame,
    *,
    domain: Tuple[float, float, float, float] = (-300, 300, -300, 300),
    dx: float = 2.0,
    zm: float = 2.0,
    z0: float = 0.1,
    h: float = 2000.0,
    include_km: bool = True,
    include_xr: bool = True,
    include_volk: bool = False,  # placeholder for future
    **extra_kwargs,
) -> pd.DataFrame:
    """
    Run multiple footprint models and compare their outputs to a reference.

    Executes several footprint models (FFP, FFP_xr, and optionally KM and Volk)
    using a common meteorological dataset and spatial domain. Each model's output
    is compared to the FFP reference using RMSE, peak distance, and 80% overlap metrics.

    Parameters
    ----------
    met : pandas.DataFrame
        DataFrame containing meteorological inputs, expected to include:
        - 'ol' : Obukhov length [m]
        - 'sigmav' : Lateral standard deviation of velocity [m/s]
        - 'ustar' : Friction velocity [m/s]
        - 'wind_dir' : Wind direction [degrees]
    domain : tuple of float, optional
        Spatial extent of the footprint domain (xmin, xmax, ymin, ymax). Default is (-300, 300, -300, 300).
    dx : float, optional
        Grid resolution in meters. Default is 2.0.
    zm : float, optional
        Measurement height [m]. Default is 2.0.
    z0 : float, optional
        Surface roughness length [m]. Default is 0.1.
    h : float, optional
        Boundary layer height [m]. Default is 2000.0.
    include_km : bool, optional
        Whether to include the KM toy model in the comparison. Default is True.
    include_xr : bool, optional
        Whether to include the `ffp_xr` model in the comparison. Default is True.
    include_volk : bool, optional
        Whether to include a Volk-style model (placeholder). Default is False.
    **extra_kwargs
        Additional arguments passed to `run_ffp`.

    Returns
    -------
    df_metrics : pandas.DataFrame
        DataFrame of comparison metrics (RMSE, peak location, 80% overlap)
        for each alternative model relative to the FFP reference.

    Notes
    -----
    - The "Volk" model is not implemented and is only a placeholder.
    - The function generates a comparison plot using `compare_footprints(plot=True)`.
    """

    ref_f, ref_grid = run_ffp(
        met, zm=zm, z0=z0, h=h, domain=domain, dx=dx, **extra_kwargs
    )

    others: Dict[str, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = {}

    if include_xr:
        f_xr, grid_xr = run_ffp_xr(met, domain=domain, dx=dx)
        others["FFP_xr"] = (f_xr, grid_xr)

    if include_km:
        f_km, grid_km = run_km(met, zm=zm, z0=z0, domain=domain, dx=dx)
        others["KM"] = (f_km, grid_km)

    # A slot for volk.py footprint (if user has a simplified wrapper)
    if include_volk:
        try:
            from volk import some_wrapper_function  # fictitious placeholder
        except ImportError:
            print("volk.py not available – skipping")
        else:
            f_v, grid_v = some_wrapper_function(met, domain=domain, dx=dx)
            others["Volk"] = (f_v, grid_v)

    df_metrics = compare_footprints(ref_f, ref_grid, others, plot=True)
    return df_metrics
