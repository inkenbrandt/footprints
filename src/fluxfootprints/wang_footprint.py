"""wang2006_footprint.py
=====================================================
Semi‑empirical flux‑footprint parameterisation (cross‑wind
integrated) after **Wang et al. 2006, JTECH 23, 1384‑1394**
plus an optional *Gaussian 2‑D reconstruction* utility.

This module provides:
    • `wang2006_fy(...)`   – 1‑D cross‑wind‑integrated footprint f(x)
    • `reconstruct_gaussian_2d(...)` – convert f(x) to f(x,y) by
      assuming a laterally Gaussian spread with σ_y(x).

The code is intended for *daytime convective boundary‑layer* (CBL)
conditions within the validity range stated by Wang et al. ( −L/h ≈ 0.01–0.1,
0.1 h ≤ zₘ ≤ 0.6 h ).  Use with caution outside that range.

"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Optional

__all__ = [
    "wang2006_fy",
    "reconstruct_gaussian_2d",
]

# ------------------------------------------------------------------
# Helper functions – coefficients from Wang et al. (2006)
# ------------------------------------------------------------------

_K1 = 0.23  # empirical coefficients published in Table 1
_K2 = 3.59  # of Wang et al. (2006) for the *adjusted* footprint
_K3 = 1.33

# The analytical base solution (their Eq. (19)) is
# f_y(x) = K1 * ( x / x_peak )^K2 * exp( -K3 * x / x_peak )
# where x_peak is the upwind distance of maximum footprint.
# Wang et al. propose (Eq. (20))
#     x_peak / h = 0.83 (zm/h)^1.16 (–L/h)^‑0.09            (1)
# for convective CBL (–L/h in 0.01…0.1).
# ------------------------------------------------------------------


def _x_peak(zm: float, h: float, L: float) -> float:
    """Return x_peak (m) as per Wang et al. Eq.​(20).

    Parameters
    ----------
    zm : float
        Measurement height above ground (m).
    h : float
        Convective boundary‑layer depth (m).
    L : float
        Monin‑Obukhov length (m).  *Negative* for unstable/convective.
    """
    if L >= 0:
        raise ValueError("Wang et al. parametrisation is for convective (L<0) only.")

    zhat = zm / h
    Lhat = -L / h  # positive dimensionless convective parameter
    return 0.83 * (zhat**1.16) * (Lhat**-0.09) * h


def wang2006_fy(
    x: np.ndarray,
    zm: float,
    h: float,
    L: float,
) -> np.ndarray:
    """Cross‑wind integrated footprint ``f_y(x)`` (m⁻¹).

    Implements Wang et al. (2006) *adjusted* CWI footprint.

    Parameters
    ----------
    x : ndarray
        1‑D array of stream‑wise distances (m, *positive upwind*).
    zm : float
        Measurement height (m).
    h : float
        Boundary‑layer height (m).
    L : float
        Obukhov length (m).  **Must be negative** for convective cases.

    Returns
    -------
    ndarray
        Footprint values f_y(x) (same shape as *x*) with \∫ f_y dx = 1.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("x must be non‑negative (positive upwind distances).")

    x_p = _x_peak(zm, h, L)
    # non‑dimensional stream‑wise coordinate
    xx = x / x_p

    fy = _K1 * (xx**_K2) * np.exp(-_K3 * xx)

    # Normalise so that ∫ fy dx = 1 (ensures exact unity despite numeric approx.)
    # Vector spacing (assume uniform spacing) for integral weight
    dx = np.gradient(x)
    integral = np.trapz(fy, x) if np.allclose(dx, dx[0]) else np.sum(fy * dx)
    fy /= integral
    return fy


# ------------------------------------------------------------------
# 2‑D Gaussian reconstruction utilities
# ------------------------------------------------------------------


def _sigma_y(
    x: np.ndarray,
    sigma_v: Optional[float] = None,
    U: Optional[float] = None,
    alpha: float = 0.3,
) -> np.ndarray:
    """Compute lateral dispersion σ_y(x).

    Formula hierarchy (choose first available):
    1. If ``sigma_v`` and ``U`` provided:  σ_y = (sigma_v / U) * x
       (ballistic growth, valid for short travel times; see textbook dispersion).
    2. Else: σ_y = α · x  (user‑tunable proportionality).

    All returns are *at least* 0.1 m to avoid zero division.
    """
    x = np.asarray(x)

    if sigma_v is not None and U is not None and U > 0:
        sig = (sigma_v / U) * x
    else:
        sig = alpha * x

    # Prevent underflow near sensor (x≈0) – set minimum width = 0.1 m
    sig = np.fmax(sig, 0.1)
    return sig


def reconstruct_gaussian_2d(
    x: np.ndarray,
    fy: np.ndarray,
    sigma_v: Optional[float] = None,
    U: Optional[float] = None,
    y_max: Optional[float] = None,
    ny: int = 201,
    alpha: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a 2‑D footprint ``f(x,y)`` from a 1‑D ``f_y(x)``.

    Assumes the lateral distribution at each stream‑wise distance ``x`` is
    *Gaussian* with standard deviation σ_y(x).

    The default dispersion law is σ_y = α·x (α≈0.3).  If turbulence statistics
    are available, pass ``sigma_v`` (lateral velocity std, m s⁻¹) and ``U``
    (mean wind speed, m s⁻¹) to use σ_y = (σ_v/U)·x.

    Parameters
    ----------
    x : ndarray
        Stream‑wise distances (m), same grid used for *fy*.
    fy : ndarray
        Cross‑wind‑integrated footprint (m⁻¹), e.g. output of `wang2006_fy`.
    sigma_v : float, optional
        Lateral velocity standard deviation (m s⁻¹).  If given with *U* overrides
        the default α‑law.
    U : float, optional
        Mean wind speed at sensor height (m s⁻¹).  Required with *sigma_v*.
    y_max : float, optional
        Half‑width of *y* domain (m).  Default = 3·max(σ_y).
    ny : int, default 201
        Number of grid points in *y* direction (symmetric about 0).
    alpha : float, default 0.3
        Proportionality constant for σ_y = α·x when *sigma_v*/*U* unavailable.

    Returns
    -------
    (X, Y, F) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        • X, Y : 2‑D coordinate grids (shape *len(x) × ny*).
        • F    : 2‑D footprint array f(x_i, y_j) (m⁻²) normalised so that
                  ∬ F dx dy = 1.
    """
    x = np.asarray(x)
    fy = np.asarray(fy)
    if x.ndim != 1 or fy.ndim != 1 or x.shape != fy.shape:
        raise ValueError("x and fy must be 1‑D arrays of the same length.")

    # Compute σ_y(x)
    sig_y = _sigma_y(x, sigma_v, U, alpha)

    # Y grid definition
    if y_max is None:
        y_max = 3.0 * np.max(sig_y)
    y = np.linspace(-y_max, y_max, ny)

    # Allocate grids
    X, Y = np.meshgrid(x, y, indexing="xy")  # shape (ny, nx)
    # Need σ_y for each column, broadcast to rows
    sig2d = sig_y[None, :]  # shape (1, nx)

    # Gaussian pdf along y for each x
    gauss = 1.0 / (np.sqrt(2.0 * np.pi) * sig2d) * np.exp(-0.5 * (Y**2) / (sig2d**2))

    # Combine with fy to get 2‑D footprint.  Note fy has units m⁻¹, gauss m⁻¹, so F m⁻².
    F = fy[None, :] * gauss

    # Normalise – ensure ∬F dx dy = 1
    dx = np.gradient(x)
    dy = y[1] - y[0]
    integral = np.sum(F * dx * dy)
    F /= integral

    return X, Y, F
