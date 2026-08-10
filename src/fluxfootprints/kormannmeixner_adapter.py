# src/fluxfootprints/kormannmeixner_adapter.py
"""
Adapter for the Kormann & Meixner (2001) analytical footprint model.
"""

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from .base_footprint_model import BaseFootprintModel
from .kormannmeixner import (
    analytical_power_law_parameters,
    length_scale_xi,
    footprint_at_points,
)


class KormannMeixnerModel(BaseFootprintModel):
    """
    Kormann & Meixner (2001) analytical footprint model adapter.

    Wraps the analytical functions in :mod:`fluxfootprints.kormannmeixner` in
    the standard :class:`BaseFootprintModel` interface. Intended to be driven
    through :func:`fluxfootprints.ffp_daily_monthly_helper.build_climatology`,
    which resolves per-timestep inputs into a DataFrame with the standardized
    column names below; the model reads those columns directly.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Must contain the following required columns:

        zm : float or Series
            Measurement height above displacement height (i.e. z - d) [m].
        z0 : float or Series
            Aerodynamic roughness length [m].
        ustar : float or Series
            Friction velocity [m s⁻¹].
        ol : float or Series
            Obukhov length [m].
        sigmav : float or Series
            Standard deviation of lateral velocity fluctuations [m s⁻¹].
        umean : float or Series
            Mean wind speed at zm [m s⁻¹].

        wind_dir : float or Series, optional
            Mean wind direction the wind blows from [deg]. When present, the
            wind-aligned footprint is rotated into the (x, y) output domain
            for that timestep; when absent, it is placed directly using the
            standard negative-upwind x convention (no rotation).
    """

    REQUIRED_COLUMNS = ["zm", "z0", "ustar", "ol", "sigmav", "umean"]

    def _validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns."""
        df = df.copy()

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.replace(-9999, np.nan)
        df = df.dropna(subset=self.REQUIRED_COLUMNS)

        return df
    
    def run(self, return_result: bool = True) -> Optional[xr.Dataset]:
        """Execute Kormann-Meixner footprint calculation."""
        self.logger.info("Starting Kormann-Meixner footprint calculation...")

        self.df = self._validate_input_df(self.df)

        # Output grid, matching the requested domain exactly.
        xmin, xmax, ymin, ymax = self.domain
        self.x = np.arange(xmin, xmax + self.dx, self.dx)
        self.y = np.arange(ymin, ymax + self.dy, self.dy)
        X_grid, Y_grid = np.meshgrid(self.x, self.y, indexing="ij")

        # Resolve power-law parameters and length scale for every timestep
        # in one vectorized call instead of looping row by row.
        zm = self.df["zm"].to_numpy(dtype=float)
        z0 = self.df["z0"].to_numpy(dtype=float)
        ustar = self.df["ustar"].to_numpy(dtype=float)
        ol = self.df["ol"].to_numpy(dtype=float)
        umean = self.df["umean"].to_numpy(dtype=float)
        sigmav = self.df["sigmav"].to_numpy(dtype=float)
        wind_dir = (
            self.df["wind_dir"].to_numpy(dtype=float)
            if "wind_dir" in self.df.columns
            else np.full(len(self.df), np.nan)
        )

        m, n, U, kappa = analytical_power_law_parameters(
            z_m=zm, z_0=z0, L=ol, u_star=ustar, u_zm=umean
        )
        xi = length_scale_xi(zm, U, kappa, m, n)

        footprint_sum = np.zeros((len(self.x), len(self.y)))

        # Process each timestep. The per-timestep footprint is evaluated
        # directly in the output frame: the fixed output grid is mapped
        # back into the wind-aligned frame (x = downwind distance, y =
        # crosswind offset) via the inverse of the rotation/reflection used
        # to place the footprint, then the closed-form KM density is
        # evaluated at those points. This avoids rebuilding a Delaunay
        # triangulation (scipy.interpolate.griddata) on every timestep and
        # is exact rather than a linear-interpolation approximation of the
        # analytical solution.
        n_valid = 0
        for i, idx in enumerate(self.df.index):
            try:
                if not np.isfinite(xi[i]):
                    raise ValueError(f"non-finite length scale xi={xi[i]!r}")

                if not np.isnan(wind_dir[i]):
                    # Wind direction available: invert the forward rotation
                    # (X_src = X cosθ - Y sinθ, Y_src = X sinθ + Y cosθ) to
                    # recover wind-aligned coordinates for each output point.
                    angle_rad = np.radians(wind_dir[i])
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    x_wind = X_grid * cos_a + Y_grid * sin_a
                    y_wind = -X_grid * sin_a + Y_grid * cos_a
                else:
                    # No wind direction: invert the standard negative-upwind
                    # placement (X_src = -X, Y_src = Y).
                    x_wind = -X_grid
                    y_wind = Y_grid

                phi_grid = footprint_at_points(
                    x_wind, y_wind, xi[i], m[i], n[i], umean[i], sigmav[i]
                )

                footprint_sum += phi_grid
                n_valid += 1

            except Exception as e:
                self.logger.warning(f"Failed to process timestep {idx}: {e}")
                continue

        if n_valid == 0:
            raise RuntimeError("No valid footprints calculated")

        # Create climatology
        self.fclim_2d = xr.DataArray(
            footprint_sum / n_valid,
            dims=("x", "y"),
            coords={"x": self.x, "y": self.y},
        )
        
        # Apply smoothing if requested
        if self.smooth_data:
            self.fclim_2d = xr.DataArray(
                gaussian_filter(self.fclim_2d.values, sigma=1.0),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )
        
        # Store results
        if return_result:
            self.results = xr.Dataset({
                "footprint_climatology": self.fclim_2d,
                "domain_x": ("x", self.x),
                "domain_y": ("y", self.y),
            })
            self.results.attrs["model"] = "Kormann-Meixner (2001)"
            self.results.attrs["n_timesteps"] = n_valid
            
            self.logger.info(f"Processed {n_valid} valid timesteps")
            return self.results
        
        return None