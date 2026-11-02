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
from .wang_footprint import _sigma_y, wang2006_fy


class WangFootprintModel(BaseFootprintModel):
    """
    Wang et al. (2006) convective footprint model adapter.

    Wraps :func:`wang2006_fy` and a Gaussian crosswind spread in the
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

        Optional column ``wind_dir`` (degrees, direction the wind blows
        FROM) orients each footprint upwind of the tower; without it the
        footprint is placed along -x. Each timestep's footprint is
        normalized to integrate to 1 over the domain before averaging.
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

        # Output grid, matching the requested domain exactly.
        xmin, xmax, ymin, ymax = self.domain
        self.x = np.arange(xmin, xmax + self.dx, self.dx)
        self.y = np.arange(ymin, ymax + self.dy, self.dy)
        X_grid, Y_grid = np.meshgrid(self.x, self.y, indexing="ij")

        # Fine 1-D upwind axis reaching the farthest grid corner. f_y(x) is
        # computed (and normalized) on it once per timestep, then sampled at
        # each grid point's upwind distance.
        r_max = float(np.hypot(np.abs(self.x).max(), np.abs(self.y).max()))
        step = min(self.dx, self.dy) / 4.0
        x_upwind = np.arange(0.0, r_max + step, step)

        wind_dir = (
            self.df["wind_dir"].to_numpy(dtype=float)
            if "wind_dir" in self.df.columns
            else np.full(len(self.df), np.nan)
        )

        footprint_sum = np.zeros((len(self.x), len(self.y)))
        n_valid = 0
        
        for i, (idx, row) in enumerate(self.df.iterrows()):
            try:
                # Project the grid into the wind-aligned frame. The wind blows
                # FROM θ (clockwise from north), so the upwind unit vector is
                # (sin θ, cos θ). Without a wind direction the footprint is
                # placed upwind along -x (wind from the west, θ = 270).
                theta = np.radians(wind_dir[i] if np.isfinite(wind_dir[i]) else 270.0)
                d = X_grid * np.sin(theta) + Y_grid * np.cos(theta)   # upwind distance
                c = X_grid * np.cos(theta) - Y_grid * np.sin(theta)   # crosswind offset

                fy = wang2006_fy(x_upwind, zm=row["zm"], h=row["h"], L=row["ol"])
                fy_grid = np.interp(d, x_upwind, fy, left=0.0, right=0.0)

                # Gaussian crosswind spread, as in reconstruct_gaussian_2d
                sig = _sigma_y(np.clip(d, 0.0, None), sigma_v=row["sigmav"], U=row["umean"])
                gauss = np.exp(-0.5 * (c / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig)
                F = np.where(d > 0, fy_grid * gauss, 0.0)

                total = F.sum() * self.dx * self.dy
                if not np.isfinite(total) or total <= 0:
                    raise ValueError("footprint falls outside the domain")
                footprint_sum += F / total
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