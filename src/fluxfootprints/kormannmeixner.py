"""kormann_meixner_footprint.py
================================================
Python implementation of the analytical flux–footprint model of Kormann & Meixner (2001).

This script provides utilities to estimate the scalar‐flux footprint of an eddy–covariance
measurement using the closed‐form solutions derived in:

    Kormann, R., & Meixner, F. X. (2001). *An analytical footprint model for non‐neutral
    stratification*. **Boundary‑Layer Meteorology, 99**, 207‑224. https://doi.org/10.1023/A:1018991015119

Only standard scientific‑Python packages are required (``numpy`` and ``scipy``).

The implementation follows the *analytical* approach described in Section 4 of the
paper to relate Monin‑Obukhov similarity profiles to the power‑law formulation used
in the footprint derivation.  If you require the more accurate (but slower)
*numerical* approach, see the companion functions in
:pyfunc:`analytical_power_law_parameters` and :pyfunc:`numerical_power_law_parameters`—the
remainder of the code is agnostic to which parameter‑estimation routine is used.

Example
-------
Estimate the cross‑wind‑integrated footprint ``f(x)`` and the full 2‑D footprint
``phi(x, y)`` for a 3‑m tower under weakly unstable conditions::

    import numpy as np
    import matplotlib.pyplot as plt
    from kormann_meixner_footprint import (
        analytical_power_law_parameters, length_scale_xi,
        crosswind_integrated_footprint, footprint_2d
    )

    # ------------------------------------------------------------------
    # Input micrometeorological parameters (replace with observations)
    z_m   = 3.0      # measurement height (m)
    z_0   = 0.03     # aerodynamic roughness length (m)
    L     = -50.0    # Obukhov length (m)  (negative ⇒ unstable)
    u_star = 0.35    # friction velocity (m s⁻¹)
    u_zm  = 2.8      # mean wind speed at z_m (m s⁻¹)
    sigma_v = 0.8    # cross‑wind velocity std‑dev (m s⁻¹)
    
    # ------------------------------------------------------------------
    # 1) Derive power‑law exponents m, n and proportionality constants U, kappa
    m, n, U, kappa = analytical_power_law_parameters(z_m, z_0, L, u_star, u_zm)

    # 2) Compute the length‑scale \xi at the measurement height
    xi = length_scale_xi(z_m, U, kappa, m, n)

    # 3) Evaluate the cross‑wind‑integrated footprint f(x)
    x = np.logspace(-1, 3.2, 500)          # 0.1–1 000 m upstream
    f_x = crosswind_integrated_footprint(x, xi, m, n)

    # 4) Evaluate the full 2‑D footprint on a coarse grid
    X, Y, phi = footprint_2d(x, np.linspace(-200, 200, 301), xi, m, n, u_zm, sigma_v)

    # 5) Plot the results
    plt.loglog(x, f_x)
    plt.xlabel("upwind distance x (m)")
    plt.ylabel("f(x, z_m)  [m⁻¹]")
    plt.title("Cross‑wind‑integrated footprint")
    plt.show()

    # Contours of the 2‑D footprint
    plt.contourf(X, Y, phi, levels=20, cmap="viridis")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Footprint density φ(x, y, z_m)")
    plt.colorbar(label="φ  [m⁻²]")
    plt.axis("equal"); plt.show()

"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaincc  # upper incomplete Γ
from typing import Tuple

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
KAPPA = 0.4  # von‑Kármán constant
PI_SQRT2 = np.sqrt(2.0 * np.pi)

# -----------------------------------------------------------------------------
# Monin–Obukhov similarity functions (Businger–Dyer relationships)
# -----------------------------------------------------------------------------

def _phi_m(z_over_L: float) -> float:
    """
    Compute the non-dimensional wind shear function φ_m(z/L).

    This function returns the stability correction for momentum
    as a function of the stability parameter z/L. It follows
    the Businger-Dyer relationships for both stable and unstable conditions.

    Parameters
    ----------
    z_over_L : float
        Stability parameter (z / L), where z is the measurement height and L is the Monin-Obukhov length.

    Returns
    -------
    float
        The value of φ_m(z/L), the stability correction for momentum.
    """
    if z_over_L >= 0.0:  # stable or neutral
        return 1.0 + 5.0 * z_over_L
    # unstable
    return (1.0 - 16.0 * z_over_L) ** -0.25


def _phi_c(z_over_L: float) -> float:
    """
    Compute the non-dimensional scalar diffusivity function φ_c(z/L).

    This function returns the stability correction for scalar transport
    (e.g., heat, vapor) as a function of the stability parameter z/L,
    using the Businger-Dyer formulation.

    Parameters
    ----------
    z_over_L : float
        Stability parameter (z / L), where z is the measurement height and L is the Monin-Obukhov length.

    Returns
    -------
    float
        The value of φ_c(z/L), the stability correction for scalar transport.
    """
    if z_over_L >= 0.0:
        return 1.0 + 5.0 * z_over_L
    return (1.0 - 16.0 * z_over_L) ** -0.5


def _psi_m(z_over_L: float) -> float:
    """
    Compute the integrated stability correction function ψ_m(z/L) for momentum.

    This function calculates the integral form of the Monin-Obukhov
    stability correction for momentum. For unstable conditions, it uses
    the formulation from Paulson (1970).

    Parameters
    ----------
    z_over_L : float
        Stability parameter (z / L), where z is the measurement height and L is the Monin-Obukhov length.

    Returns
    -------
    float
        The value of ψ_m(z/L), the integrated stability correction for momentum.
    """
    if z_over_L >= 0.0:  # stable or neutral
        return 5.0 * z_over_L
    # unstable (Paulson 1970)
    ζ = (1.0 - 16.0 * z_over_L) ** 0.25
    return (
        -2.0 * np.log((1.0 + ζ) / 2.0)
        - np.log((1.0 + ζ ** 2) / 2.0)
        + 2.0 * np.arctan(ζ)
        - np.pi / 2.0
    )

# -----------------------------------------------------------------------------
# Power‑law parameters
# -----------------------------------------------------------------------------

def analytical_power_law_parameters(
    z_m: float,
    z_0: float,
    L: float,
    u_star: float,
    u_zm: float,
) -> Tuple[float, float, float, float]:
    """Return *m*, *n*, *U*, *κ* using the *analytical* matching approach.

    Parameters
    ----------
    z_m
        Eddy‑covariance measurement height (m).
    z_0
        Aerodynamic roughness length (m).
    L
        Obukhov length (m) (negative ⇒ unstable).
    u_star
        Friction velocity (m s⁻¹).
    u_zm
        Mean wind speed at *z_m* (m s⁻¹).

    Returns
    -------
    m, n, U, kappa
        Power‑law exponents and proportionality constants for
        ``u(z) = U z**m`` and ``K(z) = kappa z**n``.
    """
    z_by_L = z_m / L if L != 0.0 else 0.0

    # Exponent for wind‑speed profile (Eq. 36)
    m = (u_star / (KAPPA * u_zm)) * _phi_m(z_by_L)

    # Exponent for eddy diffusivity profile (Eq. 36)
    if L >= 0.0:
        n = 1.0 / (1.0 + 5.0 * z_by_L)
    else:
        n = (1.0 - 24.0 * z_by_L) / (1.0 - 16.0 * z_by_L)

    # Proportionality constants by matching at z_m
    U = u_zm / (z_m ** m)
    kappa = (KAPPA * u_star / _phi_c(z_by_L)) / (z_m ** (n - 1.0))

    return m, n, U, kappa


# -----------------------------------------------------------------------------
# Core footprint equations
# -----------------------------------------------------------------------------

def length_scale_xi(z: float, U: float, kappa: float, m: float, n: float) -> float:
    """
    Calculate the characteristic footprint length-scale ξ(z).

    This function implements Eq. (19) from Kormann & Meixner (2001) to compute
    the length scale based on measurement height and atmospheric parameters.

    Parameters
    ----------
    z : float
        Measurement height above displacement height (m).
    U : float
        Mean horizontal wind speed at height z (m/s).
    kappa : float
        von Kármán constant (typically ~0.4).
    m : float
        Power law exponent for wind speed profile.
    n : float
        Power law exponent for eddy diffusivity profile.

    Returns
    -------
    float
        Characteristic length-scale ξ(z) (m).
    """
    r = 2.0 + m - n
    return (U * z ** r) / (r ** 2 * kappa)


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
        Footprint length-scale ξ(z) computed using `length_scale_xi` (m).
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
    coeff = (xi ** mu) / gamma(mu)
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
    """Return 2‑D footprint density φ(x, y, z_m).

    Parameters
    ----------
    x, y
        1‑D arrays of upstream and cross‑stream distances (m).  Positive *x* is
        up‑wind.
    xi, m, n
        Parameters returned by :pyfunc:`length_scale_xi` and
        :pyfunc:`analytical_power_law_parameters`.
    u_zm
        Mean wind speed at measurement height (m s⁻¹).
    sigma_v
        Standard deviation of cross‑wind velocity fluctuations (m s⁻¹).

    Returns
    -------
    X, Y, phi
        Meshgrids of *x*, *y* and the footprint density φ (m⁻²).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Cross‑wind integrated footprint
    f_x = crosswind_integrated_footprint(X, xi, m, n)

    # Cross‑wind dispersion σ(x) (short‑range limit)
    sigma = sigma_v * X / u_zm

    # Gaussian cross‑wind distribution Dy(x, y)
    Dy = 1.0 / (PI_SQRT2 * sigma) * np.exp(-0.5 * (Y / sigma) ** 2)

    phi = Dy * f_x
    return X, Y, phi

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
        Characteristic length-scale ξ(z) computed using `length_scale_xi` (m).
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
    return gammaincc(mu, xi / x_p)  # upper incomplete Γ / Γ(μ)


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
        Characteristic length-scale ξ(z) computed using `length_scale_xi` (m).
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
        raise ValueError("fraction must be in the open interval (0, 1)")

    # root‑solve gammaincc(mu, xi/x) = fraction  ⇒  xi/x = Q⁻¹
    r = 2.0 + m - n
    mu = (1.0 + m) / r

    def _res(x):
        return gammaincc(mu, xi / x) - fraction

    # bracket the root (x in (xi*1e‑6, xi*1e6) is usually sufficient)
    return brentq(_res, xi * 1e-6, xi * 1e6)

# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------

def _cli():  # noqa: D401 – simple CLI wrapper
    """Run a quick demo if executed as a script."""
    import argparse, sys, textwrap, matplotlib.pyplot as plt

    p = argparse.ArgumentParser(
        prog="kormann_meixner_footprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Quick command‑line driver for the Kormann–Meixner footprint model.
            Prints the effective fetch (63 % cumulative) and plots f(x).
            """
        ),
    )
    p.add_argument("z_m", type=float, help="measurement height (m)")
    p.add_argument("z_0", type=float, help="roughness length (m)")
    p.add_argument("L", type=float, help="Obukhov length (m)")
    p.add_argument("u_star", type=float, help="friction velocity (m s⁻¹)")
    p.add_argument("u_zm", type=float, help="mean wind speed at z_m (m s⁻¹)")
    p.add_argument("sigma_v", type=float, help="cross‑wind velocity SD (m s⁻¹)")
    p.add_argument("--fraction", type=float, default=0.63, help="cumulative fraction for fetch")
    args = p.parse_args()

    m, n, U, kappa = analytical_power_law_parameters(
        args.z_m, args.z_0, args.L, args.u_star, args.u_zm
    )
    xi = length_scale_xi(args.z_m, U, kappa, m, n)
    x_p = effective_fetch(args.fraction, xi, m, n)

    print(f"Power‑law exponents: m={m:.3f}, n={n:.3f}")
    print(f"Length‑scale ξ = {xi:.2f} m")
    print(f"Effective fetch (fraction={args.fraction:.2f}) = {x_p:.1f} m")

    # Plot f(x)
    x = np.logspace(-1, 3, 400)
    f_x = crosswind_integrated_footprint(x, xi, m, n)
    plt.loglog(x, f_x)
    plt.axvline(x_p, color="k", linestyle="--", label="effective fetch")
    plt.legend()
    plt.xlabel("upwind distance x (m)")
    plt.ylabel("f(x)  [m⁻¹]")
    plt.title("Kormann–Meixner footprint")
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
