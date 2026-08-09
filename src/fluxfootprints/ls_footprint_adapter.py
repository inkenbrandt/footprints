# src/fluxfootprints/ls_footprint_adapter.py
"""
Adapter for Lagrangian Stochastic footprint model.
"""

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

from .base_footprint_model import BaseFootprintModel
from .ls_footprint_model import LSFootprintConfig, BackwardLSModel


class LSFootprintModelAdapter(BaseFootprintModel):
    """
    Lagrangian Stochastic footprint model adapter.

    Wraps :class:`BackwardLSModel` in the standard :class:`BaseFootprintModel`
    interface. Intended to be driven through
    :func:`fluxfootprints.ffp_daily_monthly_helper.build_climatology`, which
    resolves per-timestep inputs into a DataFrame with the standardized
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
        wind_dir : float or Series
            Mean wind direction the wind blows from [deg].
        h : float or Series
            Boundary-layer height [m].
    n_particles : int, default 20000
        Number of stochastic particles released per timestep.
    """

    REQUIRED_COLUMNS = ["zm", "z0", "ustar", "ol", "wind_dir", "h"]

    def __init__(self, *args, n_particles: int = 20000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_particles = n_particles

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
        """Execute Lagrangian footprint calculation."""
        self.logger.info("Starting Lagrangian Stochastic footprint calculation...")

        self.df = self._validate_input_df(self.df)

        # Setup domain
        xmin, xmax, ymin, ymax = self.domain
        domain_extent = (abs(xmin), abs(ymax))

        # Initialize accumulator
        footprint_sum = None
        x_bins = None
        y_bins = None
        n_valid = 0

        # Process each timestep
        for idx, row in self.df.iterrows():
            try:
                # Create configuration directly from standardized df columns
                cfg = LSFootprintConfig(
                    zm=float(row["zm"]),
                    ustar=float(row["ustar"]),
                    L=float(row["ol"]),
                    h=float(row["h"]),
                    wind_dir_deg=float(row["wind_dir"]),
                    z0=float(row["z0"]),
                    n_particles=self.n_particles,
                    domain=domain_extent,
                    dx=self.dx,
                    dy=self.dy,
                )

                # Run model
                model = BackwardLSModel(cfg)
                model.run()

                # Get footprint
                if footprint_sum is None:
                    footprint_sum = model.footprint_2d
                    x_bins = model.x_bins
                    y_bins = model.y_bins
                else:
                    footprint_sum += model.footprint_2d

                n_valid += 1

            except Exception as e:
                self.logger.warning(f"Failed to process timestep {idx}: {e}")
                continue

        if n_valid == 0:
            raise RuntimeError("No valid footprints calculated")

        # Create coordinate arrays (bin centers)
        self.x = 0.5 * (x_bins[:-1] + x_bins[1:])
        self.y = 0.5 * (y_bins[:-1] + y_bins[1:])

        # Create climatology
        self.fclim_2d = xr.DataArray(
            footprint_sum / n_valid,
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
            self.results.attrs["model"] = "Lagrangian Stochastic"
            self.results.attrs["n_particles"] = self.n_particles
            self.results.attrs["n_timesteps"] = n_valid

            return self.results

        return None
