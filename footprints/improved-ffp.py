"""
Improved Flux Footprint Prediction (FFP) Module

This module implements the FFP model described in:
Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
A simple two-dimensional parameterisation for Flux Footprint Prediction (FFP).
Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015

This version includes improvements for better alignment with the theoretical framework.
"""

import logging
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
from scipy.ndimage import gaussian_filter

class FFPModel:
    """
    Improved implementation of the Flux Footprint Prediction model.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        domain: list = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: float = 10.0,
        dy: float = 10.0,
        nx: int = 1000,
        ny: int = 1000,
        rs: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        crop_height: float = 0.2,
        atm_bound_height: float = 2000.0,
        inst_height: float = 2.0,
        rslayer: bool = False,
        smooth_data: bool = True,
        crop: bool = False,
        verbosity: int = 2,
        logger=None,
        **kwargs,
    ):
        """
        Initialize the FFP model with configuration parameters.
        
        Args:
            df: Input DataFrame containing required meteorological data
            domain: Physical domain boundaries [xmin, xmax, ymin, ymax]
            dx, dy: Grid spacing
            nx, ny: Number of grid points
            rs: List of relative source area contributions to calculate
            crop_height: Vegetation height
            atm_bound_height: Atmospheric boundary layer height
            inst_height: Instrument height
            rslayer: Consider roughness sublayer
            smooth_data: Apply smoothing to output
            crop: Crop output to significant area
            verbosity: Logging detail level
            logger: Logger instance
        """
        # Initialize basic attributes
        self.df = df
        self.domain = domain
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.rs = rs
        self.smooth_data = smooth_data
        self.crop = crop
        self.verbosity = verbosity
        
        # Set up logger
        self.logger = logger or self._setup_logger()
        self.logger.setLevel(logging.DEBUG if verbosity > 1 else logging.INFO)
        
        # Model constants
        self.k = 0.4  # von Karman constant
        self.oln = 5000.0  # neutral stability limit
        
        # Initialize model parameters (will be updated based on stability)
        self.initialize_model_parameters()
        
        # Process input data
        self.prep_df_fields(
            crop_height=crop_height,
            inst_height=inst_height,
            atm_bound_height=atm_bound_height
        )
        
        # Set up computational domain
        self.define_domain()
        
        # Create xarray dataset
        self.create_xr_dataset()
        
        # Perform validity checks
        self.check_validity_ranges()
        
        # Handle stability regimes
        self.handle_stability_regimes()

    def _setup_logger(self):
        """Set up basic logging configuration."""
        logger = logging.getLogger('FFPModel')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def initialize_model_parameters(self):
        """Initialize model parameters with default values."""
        # Base parameters
        self.a = 1.4524
        self.b = -1.9914
        self.c = 1.4622
        self.d = 0.1359
        
        # Crosswind dispersion parameters
        self.ac = 2.17
        self.bc = 1.66
        self.cc = 20.0

    def prep_df_fields(self, crop_height: float, inst_height: float, atm_bound_height: float):
        """
        Prepare and validate input data fields.
        
        Args:
            crop_height: Vegetation height
            inst_height: Instrument height
            atm_bound_height: Atmospheric boundary layer height
        """
        # Calculate displacement height
        d_h = 10 ** (0.979 * np.log10(crop_height) - 0.154)
        
        # Add derived fields
        self.df["zm"] = inst_height - d_h  # measurement height above displacement
        self.df["h_c"] = crop_height
        self.df["z0"] = crop_height * 0.123  # roughness length
        self.df["h"] = atm_bound_height
        
        # Rename fields to standard names
        self.df = self.df.rename(columns={
            "V_SIGMA": "sigmav",
            "USTAR": "ustar",
            "wd": "wind_dir",
            "MO_LENGTH": "ol",
            "ws": "umean"
        })
        
        # Apply validity checks
        self._apply_validity_masks()
        
        # Drop invalid data
        self.df = self.df.dropna(subset=["sigmav", "wind_dir", "h", "ol"])
        self.ts_len = len(self.df)
        self.logger.debug(f"Valid input length: {self.ts_len}")

    def _apply_validity_masks(self):
        """Apply physical validity constraints to input data."""
        self.df["zm"] = np.where(self.df["zm"] <= 0.0, np.nan, self.df["zm"])
        self.df["h"] = np.where(self.df["h"] <= 10.0, np.nan, self.df["h"])
        self.df["zm"] = np.where(self.df["zm"] > self.df["h"], np.nan, self.df["zm"])
        self.df["sigmav"] = np.where(self.df["sigmav"] < 0.0, np.nan, self.df["sigmav"])
        self.df["ustar"] = np.where(self.df["ustar"] <= 0.1, np.nan, self.df["ustar"])
        self.df["wind_dir"] = np.where(
            (self.df["wind_dir"] > 360.0) | (self.df["wind_dir"] < 0.0), 
            np.nan, 
            self.df["wind_dir"]
        )

    def define_domain(self):
        """Set up the computational domain and grid."""
        # Create coordinate arrays
        if self.dx is None and self.nx is not None:
            self.x = np.linspace(self.domain[0], self.domain[1], self.nx + 1)
            self.y = np.linspace(self.domain[2], self.domain[3], self.ny + 1)
        else:
            self.x = np.arange(self.domain[0], self.domain[1] + self.dx, self.dx)
            self.y = np.arange(self.domain[2], self.domain[3] + self.dy, self.dy)
        
        # Create 2D grid
        self.xv, self.yv = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Create polar coordinate grids
        self.rho = xr.DataArray(
            np.sqrt(self.xv**2 + self.yv**2),
            dims=('x', 'y'),
            coords={'x': self.x, 'y': self.y}
        )
        self.theta = xr.DataArray(
            np.arctan2(self.yv, self.xv),
            dims=('x', 'y'),
            coords={'x': self.x, 'y': self.y}
        )
        
        # Initialize footprint grid
        self.fclim_2d = xr.zeros_like(self.rho)

    def create_xr_dataset(self):
        """Create xarray Dataset from input DataFrame."""
        self.df.index.name = 'time'
        self.ds = xr.Dataset.from_dataframe(self.df)

    def calculate_pi_groups(self):
        """
        Calculate dimensionless Π groups according to the paper.
        
        Returns:
            tuple: The four dimensionless groups
        """
        # Π1 = fy*zm
        pi_1 = self.f_2d * self.ds["zm"]
        
        # Π2 = x/zm 
        pi_2 = self.rho * np.cos(self.rotated_theta) / self.ds["zm"]
        
        # Π3 = (h - zm)/h = 1 - zm/h
        pi_3 = 1 - self.ds["zm"] / self.ds["h"]
        
        # Π4 = u(zm)/(u*k) = ln(zm/z0) - ψM
        pi_4 = self.calc_pi_4()
        
        return pi_1, pi_2, pi_3, pi_4

    def calc_pi_4(self):
        """
        Calculate Π4 with proper stability function implementation.
        
        Returns:
            xr.DataArray: Calculated Π4 values
        """
        stability_param = self.ds["zm"] / self.ds["ol"]
        
        # Calculate stability function
        psi_m = xr.where(
            self.ds["ol"] > 0,  # Stable
            -5.3 * stability_param,
            # Unstable or neutral
            xr.where(
                self.ds["ol"] < -self.oln,
                self.calc_unstable_psi(stability_param),
                0  # Neutral case
            )
        )
        
        return np.log(self.ds["zm"] / self.ds["z0"]) - psi_m

    def calc_unstable_psi(self, stability_param):
        """
        Calculate ψM for unstable conditions.
        
        Args:
            stability_param: Stability parameter zm/L
            
        Returns:
            xr.DataArray: Calculated ψM values
        """
        chi = (1 - 19 * stability_param) ** 0.25
        return (np.log((1 + chi**2)/2) + 
                2 * np.log((1 + chi)/2) - 
                2 * np.arctan(chi) + np.pi/2)

    def check_validity_ranges(self):
        """
        Check validity ranges according to equation 27 and other constraints.
        
        Returns:
            xr.Dataset: Validity masks for different constraints
        """
        validity_mask = xr.Dataset()
        
        # Height validity: 20z₀ < zm < he
        validity_mask["height_valid"] = xr.where(
            (self.ds["zm"] > 20 * self.ds["z0"]) & 
            (self.ds["zm"] < 0.8 * self.ds["h"]),
            True, False
        )
        
        # Stability validity: -15.5 ≤ zm/L
        validity_mask["stability_valid"] = xr.where(
            self.ds["zm"] / self.ds["ol"] >= -15.5,
            True, False
        )
        
        # Turbulence validity
        validity_mask["turbulence_valid"] = xr.where(
            (self.ds["ustar"] > 0.1) & 
            (self.ds["sigmav"] > 0),
            True, False
        )
        
        # Combined validity
        self.valid_footprint = validity_mask.all()
        
        return validity_mask

    def handle_stability_regimes(self):
        """
        Implement stability regime classification and parameter adjustment.
        
        Returns:
            xr.Dataset: Stability regime classifications
        """
        regimes = xr.Dataset()
        
        # Classify stability regimes
        regimes["convective"] = self.ds["ol"] < 0
        regimes["neutral"] = np.abs(self.ds["ol"]) > self.oln
        regimes["stable"] = (self.ds["ol"] > 0) & (self.ds["ol"] <= self.oln)
        
        # Adjust parameters based on regime
        self.a = xr.where(regimes["convective"], 2.930, 1.472)
        self.b = xr.where(regimes["convective"], -2.285, -1.996)
        self.c = xr.where(regimes["convective"], 2.127, 1.480)
        self.d = xr.where(regimes["convective"], -0.107, 0.169)
        
        return regimes

    def apply_paper_smoothing(self):
        """
        Apply the specific smoothing kernel from the paper.
        
        Returns:
            xr.DataArray: Smoothed footprint
        """
        skernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1, 0.4, 0.1],
            [0.05, 0.1, 0.05]
        ])
        
        smoothed = self.fclim_2d.copy()
        for _ in range(2):
            smoothed = xr.apply_ufunc(
                lambda x: signal.convolve2d(x, skernel, mode='same'),
                smoothed,
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]]
            )
        
        return smoothed

    def calculate_source_areas(self):
        """
        Calculate source areas using the paper's methodology.
        
        Returns:
            xr.Dataset: Source areas for specified relative contributions
        """
        def get_contour_levels(f, dx, dy, r):
            f_sorted = np.sort(f.values.flatten())[::-1]
            f_cumsum = np.cumsum(f_sorted) * dx * dy
            idx = np.searchsorted(f_cumsum, r)
            return f_sorted[idx] if idx < len(f_sorted) else np.nan

        source_areas = xr.Dataset()
        for r in self.rs:
            level = get_contour_levels(self.fclim_2d, self.dx, self.dy, r)
            source_areas[f"r_{int(r*100)}"] = xr.where(
                self.fclim_2d >= level,
                self.fclim_2d,
                np.nan
            )
        
        return source_areas

    def calc_xr_foot