# src/fluxfootprints/kormannmeixner_adapter.py
"""
Adapter wrapping Kormann-Meixner footprint functions in standard interface.
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
    crosswind_integrated_footprint,
    footprint_2d,
)


class KormannMeixnerModel(BaseFootprintModel):
    """
    Kormann & Meixner (2001) analytical footprint model adapter.
    
    Wraps the analytical functions in the standard BaseFootprintModel interface.
    """
    
    REQUIRED_COLUMNS = ["umean", "ustar", "z0", "ol", "sigmav"]
    
    def _validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns and handle missing data."""
        df = df.copy()
        
        # Rename columns to standard names if needed
        rename_map = {
            "WS": "umean",
            "USTAR": "ustar",
            "MO_LENGTH": "ol",
            "V_SIGMA": "sigmav",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Check required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove invalid data
        df = df.replace(-9999, np.nan)
        df = df.dropna(subset=self.REQUIRED_COLUMNS)
        
        return df
    
    def run(self, return_result: bool = True) -> Optional[xr.Dataset]:
        """Execute Kormann-Meixner footprint calculation."""
        self.logger.info("Starting Kormann-Meixner footprint calculation...")
        
        # Validate input
        self.df = self._validate_input_df(self.df)
        
        # Use directly supplied zm if available, otherwise derive from crop_height
        if self.zm is not None:
            zm = float(self.zm)
        else:
            d = 10 ** (0.979 * np.log10(self.crop_height) - 0.154)
            zm = self.inst_height - d
        
        # Setup domain
        xmin, xmax, ymin, ymax = self.domain
        self.x = np.arange(xmin, xmax + self.dx, self.dx)
        self.y = np.arange(ymin, ymax + self.dy, self.dy)
        
        # Initialize accumulator
        footprint_sum = np.zeros((len(self.x), len(self.y)))
        
        # Process each timestep
        n_valid = 0
        for idx, row in self.df.iterrows():
            try:
                # Calculate power-law parameters
                m, n, U, kappa = analytical_power_law_parameters(
                    z_m=zm,
                    z_0=row["z0"],
                    L=row["ol"],
                    u_star=row["ustar"],
                    u_zm=row["umean"],
                )
                
                # Calculate length scale
                xi = length_scale_xi(zm, U, kappa, m, n)
                
                # Calculate 2D footprint
                X, Y, phi = footprint_2d(
                    self.x,
                    self.y,
                    xi,
                    m,
                    n,
                    row["umean"],
                    row["sigmav"],
                )
                
                # Rotate by wind direction if available
                if "wind_dir" in row and not np.isnan(row["wind_dir"]):
                    angle_rad = np.radians(row["wind_dir"])
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    
                    # Rotate coordinates
                    X_rot = X * cos_a - Y * sin_a
                    Y_rot = X * sin_a + Y * cos_a
                    
                    # Interpolate to original grid
                    from scipy.interpolate import griddata
                    points = np.column_stack([X_rot.ravel(), Y_rot.ravel()])
                    values = phi.ravel()
                    X_grid, Y_grid = np.meshgrid(self.x, self.y, indexing="ij")
                    phi = griddata(
                        points,
                        values,
                        (X_grid, Y_grid),
                        method="linear",
                        fill_value=0.0
                    )
                
                footprint_sum += phi
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