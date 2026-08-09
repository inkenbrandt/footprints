# src/fluxfootprints/kormannmeixner_adapter.py
"""
Adapter for the Kormann & Meixner (2001) analytical footprint model.
"""

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from .base_footprint_model import BaseFootprintModel
from .kormannmeixner import (
    analytical_power_law_parameters,
    length_scale_xi,
    footprint_2d,
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

        # The KM crosswind-integrated footprint is only defined for strictly
        # positive downwind distance in the wind-aligned frame, so it is
        # evaluated on a separate physical grid, then placed onto the output
        # grid above: rotated by wind_dir when available, or reflected onto
        # the standard negative-upwind x convention when it is not.
        x_phys = np.arange(self.dx, max(abs(xmin), abs(xmax)) + self.dx, self.dx)
        y_phys = self.y

        footprint_sum = np.zeros((len(self.x), len(self.y)))

        # Process each timestep
        n_valid = 0
        for idx, row in self.df.iterrows():
            try:
                # Calculate power-law parameters
                m, n, U, kappa = analytical_power_law_parameters(
                    z_m=row["zm"],
                    z_0=row["z0"],
                    L=row["ol"],
                    u_star=row["ustar"],
                    u_zm=row["umean"],
                )

                # Calculate length scale
                xi = length_scale_xi(row["zm"], U, kappa, m, n)

                # Calculate 2D footprint in the wind-aligned frame (x =
                # downwind distance, y = crosswind offset).
                X, Y, phi = footprint_2d(
                    x_phys,
                    y_phys,
                    xi,
                    m,
                    n,
                    row["umean"],
                    row["sigmav"],
                )

                if "wind_dir" in row and not np.isnan(row["wind_dir"]):
                    # Rotate the wind-aligned footprint into the output frame
                    # using the meteorological wind direction.
                    angle_rad = np.radians(row["wind_dir"])
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    X_src = X * cos_a - Y * sin_a
                    Y_src = X * sin_a + Y * cos_a
                else:
                    # No wind direction available: fall back to the standard
                    # negative-upwind convention (x flipped, y unchanged).
                    X_src = -X
                    Y_src = Y

                points = np.column_stack([X_src.ravel(), Y_src.ravel()])
                values = phi.ravel()
                phi_grid = griddata(
                    points,
                    values,
                    (X_grid, Y_grid),
                    method="linear",
                    fill_value=0.0,
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