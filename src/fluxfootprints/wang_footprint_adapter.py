# src/fluxfootprints/wang_footprint_adapter.py
"""
Adapter for Wang et al. (2006) footprint model.
"""

from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from .base_footprint_model import BaseFootprintModel
from .wang_footprint import wang2006_fy, reconstruct_gaussian_2d


class WangFootprintModel(BaseFootprintModel):
    """
    Wang et al. (2006) convective footprint model adapter.

    Wraps :func:`wang2006_fy` and :func:`reconstruct_gaussian_2d` in the
    standard :class:`BaseFootprintModel` interface. Intended to be driven
    through :func:`fluxfootprints.ffp_daily_monthly_helper.build_climatology`,
    which resolves per-timestep inputs into a DataFrame with the standardized
    column names below; the model reads those columns directly.

    Note: Valid only for daytime convective conditions (Obukhov length < 0).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Must contain the following required columns:

        zm : float or Series
            Measurement height above displacement height (i.e. z - d) [m].
        h : float or Series
            Boundary-layer height [m].
        ol : float or Series
            Obukhov length [m]. Must be negative (convective) per timestep.
        sigmav : float or Series
            Standard deviation of lateral velocity fluctuations [m s⁻¹].
        umean : float or Series
            Mean wind speed at zm [m s⁻¹].
    """

    REQUIRED_COLUMNS = ["zm", "h", "ol", "sigmav", "umean"]

    def _validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns."""
        df = df.copy()

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.replace(-9999, np.nan)
        df = df.dropna(subset=self.REQUIRED_COLUMNS)

        # Filter for convective conditions only
        df = df[df["ol"] < 0]

        if len(df) == 0:
            raise ValueError("Wang model requires convective conditions (L < 0)")

        return df
    
    def run(self, return_result: bool = True) -> Optional[xr.Dataset]:
        """Execute Wang footprint calculation."""
        self.logger.info("Starting Wang et al. (2006) footprint calculation...")
        
        self.df = self._validate_input_df(self.df)

        # Setup x domain (positive upwind)
        xmin, xmax, ymin, ymax = self.domain
        self.x = np.arange(0, abs(xmin) + self.dx, self.dx)
        
        # Initialize accumulator
        footprint_sum_2d = None
        n_valid = 0
        
        for idx, row in self.df.iterrows():
            try:
                # Calculate 1D footprint
                fy = wang2006_fy(
                    self.x,
                    zm=row["zm"],
                    h=row["h"],
                    L=row["ol"],
                )
                
                # Reconstruct 2D
                X, Y, F = reconstruct_gaussian_2d(
                    self.x,
                    fy,
                    sigma_v=row["sigmav"],
                    U=row["umean"],
                    y_max=abs(ymax),
                    ny=int((ymax - ymin) / self.dy) + 1,
                )
                
                if footprint_sum_2d is None:
                    footprint_sum_2d = F
                    self.y = Y[:, 0]  # Extract y coordinates
                else:
                    footprint_sum_2d += F
                
                n_valid += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process timestep {idx}: {e}")
                continue
        
        if n_valid == 0:
            raise RuntimeError("No valid footprints calculated")
        
        # Flip x to match standard convention (negative upwind)
        self.x = -self.x[::-1]
        footprint_sum_2d = footprint_sum_2d[:, ::-1]
        
        # Create climatology
        self.fclim_2d = xr.DataArray(
            footprint_sum_2d.T / n_valid,
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
        
        if return_result:
            self.results = xr.Dataset({
                "footprint_climatology": self.fclim_2d,
                "domain_x": ("x", self.x),
                "domain_y": ("y", self.y),
            })
            self.results.attrs["model"] = "Wang et al. (2006)"
            self.results.attrs["note"] = "Valid for convective conditions only"
            self.results.attrs["n_timesteps"] = n_valid
            
            return self.results
        
        return None