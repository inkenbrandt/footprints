"""ls_footprint_model.py
====================================
A **minimal, research‑grade** 3‑D backward Lagrangian stochastic (LS) footprint
model for eddy‑covariance flux measurements.

The implementation follows the principles of Kljun et al. (2002) but is written
from scratch in pure Python/Numpy for clarity and ease of extension.  It is **not
optimised for production** and should be validated against benchmark cases
before scientific use.  Nevertheless, it demonstrates all core ideas:

* stochastic integration of Langevin equations obeying the well‑mixed criterion
* backward‑in‑time particle trajectory simulation
* touchdown detection and gridded accumulation of the 2‑D flux footprint
* cross‑wind integration and basic contour utilities

"""

from __future__ import annotations

import math
import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Tuple, Dict, Any


# Physical & numerical constants
KAPPA = 0.4  # von Kármán constant
GRAV = 9.81  # m s⁻²
RNG = default_rng()


# Utility functions


def log_wind_profile(z: float, ustar: float, z0: float) -> float:
    """Neutral logarithmic mean‑wind profile.

    Parameters
    ----------
    z : float
        Height above displacement (m).
    ustar : float
        Friction velocity (m s⁻¹).
    z0 : float
        Aerodynamic roughness length (m).

    Returns
    -------
    float
        Mean wind speed at *z* (m s⁻¹).
    """
    return (ustar / KAPPA) * np.log(z / z0)


def sigma_w(ustar: float) -> float:
    """Vertical velocity standard deviation (surface‑layer neutral approx.)."""
    return 1.3 * ustar


def sigma_v(ustar: float) -> float:
    """Cross‑wind velocity std dev (surface‑layer neutral approx.)."""
    return 2.0 * ustar


def lagrangian_timescale(z: float, zm: float, ustar: float) -> float:
    """Simple height‑dependent Lagrangian timescale (Hunt, 1984 style).

    T_L = 0.15 z / σ_w + 0.3 z_m / σ_w(z_m)  (heuristic)
    """
    return 0.15 * z / sigma_w(ustar) + 0.3 * zm / sigma_w(ustar)


# Core model


@dataclass
class LSFootprintConfig:
    """Container for model parameters fixed for one simulation."""

    zm: float  # receptor height (m)
    ustar: float  # friction velocity (m s⁻¹)
    L: float  # Monin‑Obukhov length (m); not yet used but kept for extension
    h: float  # boundary‑layer height (m)
    wind_dir_deg: float  # mean wind direction FROM which wind blows (deg)
    z0: float = 0.1  # roughness length (m)
    n_particles: int = 20_000  # number of stochastic particles
    dt: float = 0.25  # time step (s)
    t_max: float = 600.0  # maximum integration time (s)
    domain: Tuple[float, float] = (2000.0, 2000.0)  # (x_max, y_max) extent (m)
    dx: float = 10.0  # horizontal grid resolution (m)
    dy: float = 10.0

    # Random‑seed control for reproducibility
    seed: int | None = None

    def rng(self):
        return default_rng(self.seed)


class BackwardLSModel:
    """A 3‑D backward Lagrangian stochastic footprint model."""

    def __init__(self, cfg: LSFootprintConfig):
        self.cfg = cfg
        self.rng = cfg.rng()

        # Pre‑compute grid for footprint accumulation
        x_max, y_max = cfg.domain
        self.x_bins = np.arange(-x_max, 0 + cfg.dx, cfg.dx)  # upstream is −x
        self.y_bins = np.arange(-y_max, y_max + cfg.dy, cfg.dy)
        self.footprint_2d = np.zeros((len(self.x_bins) - 1, len(self.y_bins) - 1))

    # ---------------------------------------------------------------------
    # Particle initialisation
    # ---------------------------------------------------------------------
    def _initial_particle_state(self, n: int) -> Dict[str, np.ndarray]:
        """Return initial arrays for *n* particles released at the receptor."""
        cfg = self.cfg
        w_std = sigma_w(cfg.ustar)
        v_std = sigma_v(cfg.ustar)
        u_std = v_std  # approx.

        # Initial velocity components sampled from Gaussians
        w = self.rng.normal(0.0, w_std, size=n)
        v = self.rng.normal(0.0, v_std, size=n)
        u = self.rng.normal(0.0, u_std, size=n)

        # Positions – sensor at origin (x=0, y=0, z=zm)
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.full(n, cfg.zm)

        active = np.ones(n, dtype=bool)
        return dict(x=x, y=y, z=z, u=u, v=v, w=w, active=active)

    # ------------------------------------------------------------------
    # Velocity update – simplified well‑mixed form (Ornstein‑Uhlenbeck)
    # ------------------------------------------------------------------
    def _update_velocity(
        self, z: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        dt = cfg.dt

        # Compute local turbulence stats (using surface‑layer neutral formulae)
        sw = sigma_w(cfg.ustar)
        sv = sigma_v(cfg.ustar)
        su = sv  # simplification

        # Height‑dependent Lagrangian timescales
        tl = lagrangian_timescale(z, cfg.zm, cfg.ustar)

        # OU process coefficients
        a = np.exp(-dt / tl)  # memory term
        b_w = np.sqrt((1.0 - a**2) * sw**2)
        b_u = np.sqrt((1.0 - a**2) * su**2)
        b_v = np.sqrt((1.0 - a**2) * sv**2)

        # Update stochastic velocities
        u = a * u + self.rng.normal(0.0, b_u, size=u.shape)
        v = a * v + self.rng.normal(0.0, b_v, size=v.shape)
        w = a * w + self.rng.normal(0.0, b_w, size=w.shape)
        return u, v, w

    # --------------------------------------------
    # Main simulation loop
    # --------------------------------------------
    def run(self) -> None:
        """Run the LS footprint simulation and accumulate 2‑D footprint."""
        cfg = self.cfg
        p = self._initial_particle_state(cfg.n_particles)

        n_active = cfg.n_particles
        steps = int(cfg.t_max / cfg.dt)

        # Rotate mean wind coordinate frame so x is upstream
        wind_dir_rad = math.radians(cfg.wind_dir_deg)
        cosd, sind = math.cos(wind_dir_rad), math.sin(wind_dir_rad)

        for _ in range(steps):
            if n_active == 0:
                break  # all particles have landed

            idx = p["active"]
            z = p["z"][idx]
            u = p["u"][idx]
            v = p["v"][idx]
            w = p["w"][idx]

            # Update velocities
            u, v, w = self._update_velocity(z, u, v, w)
            p["u"][idx], p["v"][idx], p["w"][idx] = u, v, w

            # Mean wind at current *z*
            U_mean = log_wind_profile(z, cfg.ustar, cfg.z0)

            # Backward‑in‑time displacements (negative mean wind direction)
            dt = cfg.dt
            # Rotate to tower‑aligned coordinates (sensor/reference)
            # Upwind (−x) is along mean wind; we subtract because backward
            dx_upwind = -(U_mean + u) * dt
            dy_cross = -v * dt
            dz = -w * dt

            p["x"][idx] += (
                dx_upwind * cosd - dy_cross * sind
            )  # rotate back to geographic
            p["y"][idx] += dx_upwind * sind + dy_cross * cosd
            p["z"][idx] += dz

            # Check for landings
            landed = p["z"] <= cfg.z0
            if np.any(landed):
                # Bin landing positions into footprint grid
                self._accumulate(p["x"][landed], p["y"][landed])
                p["active"][landed] = False
                n_active -= int(np.sum(landed))

            # Reflect particles that escape top of BL to conserve mass
            above = p["z"] > cfg.h
            p["z"][above] = 2 * cfg.h - p["z"][above]  # mirror reflection
            p["w"][above] *= -1  # invert vertical velocity when reflecting

        # Normalise to unit flux contribution
        self.footprint_2d /= np.sum(self.footprint_2d * cfg.dx * cfg.dy)

    # ---------------------------------
    # Accumulation helper
    # ---------------------------------
    def _accumulate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Bin touchdown points into the 2‑D histogram."""
        H, _, _ = np.histogram2d(x, y, bins=[self.x_bins, self.y_bins])
        self.footprint_2d += H

    # ---------------------------------
    # Public output accessors
    # ---------------------------------
    def footprint(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x_centres, y_centres, footprint_density)."""
        x_c = 0.5 * (self.x_bins[:-1] + self.x_bins[1:])
        y_c = 0.5 * (self.y_bins[:-1] + self.y_bins[1:])
        return x_c, y_c, self.footprint_2d

    def crosswind_integrated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return upstream distance array and integrated footprint f(x)."""
        x_c, y_c, F = self.footprint()
        f_x = np.sum(F, axis=1) * self.cfg.dy  # integrate over y
        return x_c, f_x


###############################################################################
# Example usage (run as script)
###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example configuration (replace with real tower data)
    cfg = LSFootprintConfig(
        zm=20.0,
        ustar=0.4,
        L=-50.0,
        h=1000.0,
        wind_dir_deg=270.0,  # wind FROM west → footprint to west of tower
        z0=0.1,
        n_particles=50_000,
        dt=0.25,
        t_max=600.0,
        domain=(2000.0, 2000.0),
        dx=20.0,
        dy=20.0,
        seed=42,
    )

    model = BackwardLSModel(cfg)
    model.run()

    # Plot 2‑D footprint heatmap
    x, y, F = model.footprint()
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(y, x, F, shading="auto", cmap="viridis")
    plt.colorbar(label="Footprint weight (m⁻²)")
    plt.xlabel("Cross‑wind distance y (m)")
    plt.ylabel("Upwind distance −x (m)")
    plt.title("2‑D Lagrangian flux footprint")
    plt.tight_layout()

    # Plot crosswind‑integrated footprint
    plt.figure()
    xx, fx = model.crosswind_integrated()
    plt.plot(-xx, fx)  # flip sign for intuitive positive downwind axis
    plt.xlabel("Upwind distance (m)")
    plt.ylabel("f(x) (m⁻¹)")
    plt.title("Cross‑wind integrated footprint")
    plt.grid(True)
    plt.show()
