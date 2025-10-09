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
    
    Wraps the BackwardLSModel in the standard interface.
    """
    
    REQUIRED_COLUMNS = ["ustar", "ol", "wind_dir"]
    
    def __init__(self, *args, n_particles: int = 20000, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_particles = n_particles
    
    def _validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns."""
        df = df.copy()
        
        # Rename columns
        rename_map = {
            "USTAR": "ustar",
            "MO_LENGTH": "ol",
            "WD": "wind_dir",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Check required columns
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
        
        # Calculate displacement and measurement height
        d = 10 ** (0.979 * np.log10(self.crop_height) - 0.154)
        zm = self.inst_height - d
        z0 = self.crop_height * 0.123
        
        # Setup domain
        xmin, xmax, ymin, ymax = self.domain
        domain_extent = (abs(xmin), abs(ymax))
        
        # Initialize accumulator
        footprint_sum = None
        x_bins = None
        y_bins = None
        n_valid = 0
        
        # Process each timestep (or use mean conditions)
        for idx, row in self.df.iterrows():
            try:
                # Create configuration
                cfg = LSFootprintConfig(
                    zm=zm,
                    ustar=row["ustar"],
                    L=row["ol"],
                    h=self.atm_bound_height,
                    wind_dir_deg=row["wind_dir"] if "wind_dir" in row else 270.0,
                    z0=z0,
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