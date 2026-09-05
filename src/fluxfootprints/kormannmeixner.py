"""
kormann_meixner_footprint.py
================================================
Python implementation of the analytical flux-footprint model of Kormann & Meixner (2001).

This script provides utilities to estimate the scalar-flux footprint of an eddy-covariance
measurement using the closed-form solutions derived in:

    Kormann, R., & Meixner, F. X. (2001). *An analytical footprint model for non-neutral
    stratification*. **Boundary-Layer Meteorology, 99**, 207-224. https://doi.org/10.1023/A:1018991015119

Only standard scientific-Python packages are required (``numpy`` and ``scipy``).

The implementation follows the *analytical* approach described in Section 4 of the
paper to relate Monin-Obukhov similarity profiles to the power-law formulation used
in the footprint derivation.  If you require the more accurate (but slower)
*numerical* approach, see the companion functions in
:func:`analytical_power_law_parameters` and :func:`numerical_power_law_parameters`â€”the
remainder of the code is agnostic to which parameter-estimation routine is used.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaincc  # upper incomplete Î“
from typing import Tuple

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
KAPPA = 0.4  # von-KÃ¡rmÃ¡n constant
PI_SQRT2 = np.sqrt(2.0 * np.pi)

# -----------------------------------------------------------------------------
# Monin-Obukhov similarity functions (Businger-Dyer relationships)
# -----------------------------------------------------------------------------


def _descalar(a: np.ndarray) -> float | np.ndarray:
    """Return a Python float for 0-d arrays, else the array unchanged.

    Lets the vectorized stability/power-law functions below accept either
    scalars or arrays and hand back the matching shape, so a single
    implementation serves both a one-off call and a whole-column call over
    a DataFrame.
    """
    a = np.asarray(a)
    return a.item() if a.ndim == 0 else a


def _phi_m(z_over_L: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the non-dimensional wind shear function Ï†_m(z/L).

    This function returns the stability correction for momentum
    as a function of the stability parameter z/L. It follows
    the Businger-Dyer relationships for both stable and unstable conditions.
    Accepts scalars or arrays; both branches are evaluated elementwise and
    selected with :func:`numpy.where`, so this is safe to call on a whole
    column of z/L values at once.

    Parameters
    ----------
    z_over_L : float or np.ndarray
        Stability parameter (z / L), where z is the measurement height and L is the Monin-Obukhov length.

    Returns
    -------
    float or np.ndarray
        The value of Ï†_m(z/L), the stability correction for momentum.
    """
    z_over_L = np.asarray(z_over_L, dtype=float)
    with np.errstate(invalid="ignore"):
        stable = 1.0 + 5.0 * z_over_L
        unstable = (1.0 - 16.0 * z_over_L) ** -0.25
    return _descalar(np.where(z_over_L >= 0.0, stable, unstable))


def _phi_c(z_over_L: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the non-dimensional scalar diffusivity function Ï†_c(z/L).

    This function returns the stability correction for scalar transport
    (e.g., heat, vapor) as a function of the stability parameter z/L,
    using the Businger-Dyer formulation. Accepts scalars or arrays; see
    :func:`_phi_m` for the elementwise-selection approach.

    Parameters
    ----------
    z_over_L : float or np.ndarray
        Stability parameter (z / L), where z is the measurement height and L is the Monin-Obukhov length.

    Returns
    -------
    float or np.ndarray
        The value of Ï†_c(z/L), the stability correction for scalar transport.
    """
    z_over_L = np.asarray(z_over_L, dtype=float)
    with np.errstate(invalid="ignore"):
        stable = 1.0 + 5.0 * z_over_L
        unstable = (1.0 - 16.0 * z_over_L) ** -0.5
    return _descalar(np.where(z_over_L >= 0.0, stable, unstable))


def _psi_m(z_over_L: float | np.ndarray) -> float | np.ndarray:
    """
    Compute the integrated stability correction function Ïˆ_m(z/L) for momentum.

    This function calculates the integral form of the Monin-Obukhov
    stability correction for momentum. For unstable conditions, it uses
    the formulation from Paulson (1970). Accepts scalars or arrays; see
    :func:`_phi_m` for the elementwise-selection approach.

    Parameters
    ----------
    z_over_L : float or np.ndarray
        Stability parameter (z / L), where z is the measurement height and L is the Monin-Obukhov length.

    Returns
    -------
    float or np.ndarray
        The value of Ïˆ_m(z/L), the integrated stability correction for momentum.
    """
    z_over_L = np.asarray(z_over_L, dtype=float)
    with np.errstate(invalid="ignore"):
        stable = 5.0 * z_over_L
        # unstable (Paulson 1970)
        zeta = (1.0 - 16.0 * z_over_L) ** 0.25
        unstable = (
            -2.0 * np.log((1.0 + zeta) / 2.0)
            - np.log((1.0 + zeta**2) / 2.0)
            + 2.0 * np.arctan(zeta)
            - np.pi / 2.0
        )
    return _descalar(np.where(z_over_L >= 0.0, stable, unstable))


# -----------------------------------------------------------------------------
# Power-law parameters
# -----------------------------------------------------------------------------


def analytical_power_law_parameters(
    z_m: float | np.ndarray,
    z_0: float | np.ndarray,
    L: float | np.ndarray,
    u_star: float | np.ndarray,
    u_zm: float | np.ndarray,
) -> Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Return *m*, *n*, *U*, *Îº* using the *analytical* matching approach.

    Accepts scalars or same-shaped arrays for every argument, so a whole
    column of per-timestep inputs can be resolved in one vectorized call
    instead of looping row by row.

    Parameters
    ----------
    z_m
        Eddy-covariance measurement height (m).
    z_0
        Aerodynamic roughness length (m).
    L
        Obukhov length (m) (negative â‡’ unstable).
    u_star
        Friction velocity (m sâ»Â¹).
    u_zm
        Mean wind speed at *z_m* (m sâ»Â¹).

    Returns
    -------
    m, n, U, kappa
        Power-law exponents and proportionality constants for
        ``u(z) = U z**m`` and ``K(z) = kappa z**n``.
    """
    z_m = np.asarray(z_m, dtype=float)
    L = np.asarray(L, dtype=float)
    u_star = np.asarray(u_star, dtype=float)
    u_zm = np.asarray(u_zm, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_by_L = np.where(L != 0.0, z_m / np.where(L != 0.0, L, 1.0), 0.0)

        # Exponent for wind-speed profile (Eq. 36)
        m = (u_star / (KAPPA * u_zm)) * _phi_m(z_by_L)

        # Exponent for eddy diffusivity profile (Eq. 36)
        n_stable = 1.0 / (1.0 + 5.0 * z_by_L)
        n_unstable = (1.0 - 24.0 * z_by_L) / (1.0 - 16.0 * z_by_L)
        n = np.where(L >= 0.0, n_stable, n_unstable)

        # Proportionality constants by matching at z_m
        U = u_zm / (z_m**m)
        kappa = (KAPPA * u_star / _phi_c(z_by_L)) / (z_m ** (n - 1.0))

    return _descalar(m), _descalar(n), _descalar(U), _descalar(kappa)


# -----------------------------------------------------------------------------
# Core footprint equations
# -----------------------------------------------------------------------------


def length_scale_xi(z: float, U: float, kappa: float, m: float, n: float) -> float:
    """
    Calculate the characteristic footprint length-scale Î¾(z).

    This function implements Eq. (19) from Kormann & Meixner (2001) to compute
    the length scale based on measurement height and atmospheric parameters.

    Parameters
    ----------
    z : float
        Measurement height above displacement height (m).
    U : float
        Mean horizontal wind speed at height z (m/s).
    kappa : float
        von KÃ¡rmÃ¡n constant (typically ~0.4).
    m : float
        Power law exponent for wind speed profile.
    n : float
        Power law exponent for eddy diffusivity profile.

    Returns
    -------
    float
        Characteristic length-scale Î¾(z) (m).
    """
    r = 2.0 + m - n
    return (U * z**r) / (r**2 * kappa)


def crosswind_integrated_footprint(
    x: np.ndarray | float,
    xi: float,
    m: float,
    n: float,
) -> np.ndarray | float:
    """
    Compute the cross-wind-integrated footprint f(x, z).

    This function implements Eq. (21) from Kormann & Meixner (2001), which
    describes the probability density function of source area contributions
    in the along-wind direction, integrated over the cross-wind direction.

    Parameters
    ----------
    x : float or np.ndarray
        Downwind distance(s) from the tower (m).
    xi : float
        Footprint length-scale Î¾(z) computed using `length_scale_xi` (m).
    m : float
        Power law exponent for wind speed profile.
    n : float
        Power law exponent for eddy diffusivity profile.

    Returns
    -------
    float or np.ndarray
        Cross-wind-integrated footprint value(s) at distance x.
    """
    r = 2.0 + m - n
    mu = (1.0 + m) / r
    coeff = (xi**mu) / gamma(mu)
    x = np.asarray(x)
    return coeff * x ** (-(1.0 + mu)) * np.exp(-xi / x)


def footprint_2d(
    x: np.ndarray,
    y: np.ndarray,
    xi: float,
    m: float,
    n: float,
    u_zm: float,
    sigma_v: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2-D footprint density Ï†(x, y, z_m).

    Parameters
    ----------
    x, y
        1-D arrays of upstream and cross-stream distances (m).  Positive *x* is
        up-wind.
    xi, m, n
        Parameters returned by :func:`length_scale_xi` and
        :func:`analytical_power_law_parameters`.
    u_zm
        Mean wind speed at measurement height (m sâ»Â¹).
    sigma_v
        Standard deviation of cross-wind velocity fluctuations (m sâ»Â¹).

    Returns
    -------
    X, Y, phi
        Meshgrids of *x*, *y* and the footprint density Ï† (mâ»Â²).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Cross-wind integrated footprint
    f_x = crosswind_integrated_footprint(X, xi, m, n)

    # Cross-wind dispersion Ïƒ(x) (short-range limit)
    sigma = sigma_v * X / u_zm

    # Gaussian cross-wind distribution Dy(x, y)
    Dy = 1.0 / (PI_SQRT2 * sigma) * np.exp(-0.5 * (Y / sigma) ** 2)

    phi = Dy * f_x
    return X, Y, phi


def footprint_at_points(
    x: np.ndarray,
    y: np.ndarray,
    xi: float,
    m: float,
    n: float,
    u_zm: float,
    sigma_v: float,
) -> np.ndarray:
    """Evaluate the closed-form 2-D footprint density directly at given points.

    Implements the same Eq. (19)/(21) formulation as
    :func:`crosswind_integrated_footprint`/:func:`footprint_2d`, but
    takes ``x``/``y`` as same-shaped arrays of arbitrary points (e.g. an
    already-rotated output grid) rather than building the density on the
    outer product of separate 1-D wind-aligned axes. This lets a footprint
    grid in a rotated/translated frame be filled by direct evaluation
    instead of computing it on a wind-aligned grid and then interpolating
    (e.g. via :func:`scipy.interpolate.griddata`) onto the target frame.

    Parameters
    ----------
    x, y
        Arrays of downwind distance and crosswind offset (m) in the
        wind-aligned frame, same shape. Points with ``x <= 0`` are upwind of
        the sensor, where the model is undefined, and are assigned zero
        density.
    xi, m, n
        Parameters returned by :func:`length_scale_xi` and
        :func:`analytical_power_law_parameters`.
    u_zm
        Mean wind speed at measurement height (m sâ»Â¹).
    sigma_v
        Standard deviation of cross-wind velocity fluctuations (m sâ»Â¹).

    Returns
    -------
    np.ndarray
        Footprint density Ï† (mâ»Â²), same shape as ``x``/``y``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = x > 0.0

    r = 2.0 + m - n
    mu = (1.0 + m) / r
    coeff = (xi**mu) / gamma(mu)

    # Substitute a safe placeholder for invalid (x <= 0) points so the
    # closed-form expressions don't raise divide/invalid warnings; those
    # entries are masked out below regardless of what they evaluate to.
    x_safe = np.where(valid, x, 1.0)
    f_x = coeff * x_safe ** (-(1.0 + mu)) * np.exp(-xi / x_safe)

    sigma = sigma_v * x_safe / u_zm
    Dy = 1.0 / (PI_SQRT2 * sigma) * np.exp(-0.5 * (y / sigma) ** 2)

    return np.where(valid, Dy * f_x, 0.0)


# -----------------------------------------------------------------------------
# Fetch and auxiliary functions
# -----------------------------------------------------------------------------


def cumulative_fetch(x_p: float, xi: float, m: float, n: float) -> float:
    """
    Calculate the cumulative fetch P(x_p), the fraction of flux originating upwind of x_p.

    Implements Eq. (29) from Kormann & Meixner (2001), returning the cumulative
    contribution to the flux footprint up to a specified downwind distance.

    Parameters
    ----------
    x_p : float
        Downwind distance from the tower (m) at which the cumulative flux contribution is evaluated.
    xi : float
        Characteristic length-scale Î¾(z) computed using `length_scale_xi` (m).
    m : float
        Power law exponent for wind speed profile.
    n : float
        Power law exponent for eddy diffusivity profile.

    Returns
    -------
    float
        Fraction of total flux (between 0 and 1) originating upwind of x_p.
    """
    r = 2.0 + m - n
    mu = (1.0 + m) / r
    return gammaincc(mu, xi / x_p)  # upper incomplete Î“ / Î“(Î¼)


def effective_fetch(fraction: float, xi: float, m: float, n: float) -> float:
    """
    Invert the cumulative fetch function to determine the fetch distance x_p for a given flux fraction.

    Solves for x_p such that `cumulative_fetch(x_p) = fraction`, which identifies
    the distance upwind from which a specified fraction of the total flux originates.

    Parameters
    ----------
    fraction : float
        Desired cumulative flux contribution (must be in the open interval (0, 1)).
    xi : float
        Characteristic length-scale Î¾(z) computed using `length_scale_xi` (m).
    m : float
        Power law exponent for wind speed profile.
    n : float
        Power law exponent for eddy diffusivity profile.

    Returns
    -------
    float
        Effective fetch distance x_p (m) upwind of the sensor that contributes the given flux fraction.

    Raises
    ------
    ValueError
        If `fraction` is not in the open interval (0, 1).
    """
    from scipy.optimize import brentq

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in the open interval (0, 1)")

    # root-solve gammaincc(mu, xi/x) = fraction  â‡’  xi/x = Qâ»Â¹
    r = 2.0 + m - n
    mu = (1.0 + m) / r

    def _res(x):
        return gammaincc(mu, xi / x) - fraction

    # bracket the root (x in (xi*1e-6, xi*1e6) is usually sufficient)
