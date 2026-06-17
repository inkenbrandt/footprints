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
import matplotlib.pyplot as plt
import traceback
from .base_footprint_model import BaseFootprintModel

class FFPModel(BaseFootprintModel):
    """
    Flux Footprint Prediction (FFP) model based on Kljun et al. (2015).

    This class implements an improved version of the flux footprint model
    for determining the source area of scalar fluxes measured by
    eddy covariance towers. It uses atmospheric stability, wind parameters,
    and terrain configurations to calculate spatial footprints.

    Attributes
    ----------
    REQUIRED_COLUMNS : list of str
        Names of required columns in the input DataFrame.

    df : pandas.DataFrame
        Copy of the input DataFrame used for modeling.

    domain : list of float
        Spatial domain defined as [xmin, xmax, ymin, ymax].

    fclim_2d : xarray.DataArray or None
        Final footprint climatology after all processing.

    f_2d : xarray.DataArray or None
        Instantaneous 2D footprint estimate.

    logger : logging.Logger
        Logger instance for model-level debugging.
    """

    REQUIRED_COLUMNS = [
        "sigmav",  # Standard deviation of lateral velocity fluctuations
        "ustar",  # Friction velocity
        "ol",  # Obukhov length
        "wind_dir",  # Wind direction
        "umean",  # Wind speed
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        domain: list = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: float = 10.0,
        dy: float = 10.0,
        nx: int = 1000,
        ny: int = 1000,
        rs: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        crop_height: float = 0.2,
        atm_bound_height: float = 2000.0,
        inst_height: float = 2.0,
        zm: Optional[float] = None,
        z0: Optional[float] = None,
        rslayer: bool = False,
        smooth_data: bool = True,
        crop: bool = False,
        verbosity: int = 2,
        logger=None,
        **kwargs,
    ):
        """
        Initialize the FFP model with input data and configuration.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with required meteorological columns including:
            'V_SIGMA', 'USTAR', 'MO_LENGTH', 'WD', 'WS'.

        domain : list of float, optional
            Footprint domain in the format [xmin, xmax, ymin, ymax].

        dx : float, optional
            Grid spacing in the x-direction (meters). Default is 10.0.

        dy : float, optional
            Grid spacing in the y-direction (meters). Default is 10.0.

        nx : int, optional
            Number of grid points in the x-direction. Default is 1000.

        ny : int, optional
            Number of grid points in the y-direction. Default is 1000.

        rs : list of float, optional
            Relative source area contributions to evaluate. Values must be between 0 and 1.

        zm : float, optional
            Effective measurement height above the zero-plane displacement height
            (i.e. ``z_instrument - d``) [m].  When provided together with ``z0``,
            these values are used directly and ``crop_height`` / ``inst_height``
            are ignored for the height derivation.

        z0 : float, optional
            Aerodynamic roughness length [m].  Must be supplied together with
            ``zm`` when bypassing the ``crop_height`` derivation.

        crop_height : float, optional
            Height of vegetation or surface roughness (m). Used to derive ``zm``
            and ``z0`` when those are not provided directly. Default is 0.2.

        atm_bound_height : float, optional
            Height of the atmospheric boundary layer (m). Must be > 10. Default is 2000.0.

        inst_height : float, optional
            Height of the measurement instrument (m). Used to derive ``zm`` when
            ``zm`` is not provided directly. Must be > crop_height. Default is 2.0.

        rslayer : bool, optional
            If True, apply roughness sublayer corrections. Default is False.

        smooth_data : bool, optional
            Whether to apply smoothing to the footprint climatology. Default is True.

        crop : bool, optional
            If True, crop output to significant areas. Default is False.

        verbosity : int, optional
            Verbosity level for logging (0=none, 1=info, 2=debug). Default is 2.

        logger : logging.Logger, optional
            Custom logger. If None, a new logger is created.

        kwargs : dict
            Additional keyword arguments for future extensions.

        Raises
        ------
        ValueError
            If any input parameters are invalid or inconsistent.
        """
        # Validate atmospheric boundary layer height (always required)
        if atm_bound_height <= 10:
            raise ValueError("atm_bound_height must be > 10m")

        # Resolve effective measurement height and roughness length
        if zm is not None and z0 is not None:
            # Direct inputs take priority over crop_height / inst_height
            zm_eff = float(zm)
            z0_eff = float(z0)
            if zm_eff <= 0:
                raise ValueError("zm must be positive")
            if z0_eff <= 0:
                raise ValueError("z0 must be positive")
        elif zm is not None or z0 is not None:
            raise ValueError(
                "Both zm and z0 must be provided together. "
                "Supply neither (use crop_height + inst_height) or both."
            )
        else:
            # Backward-compatible path: derive from crop_height and inst_height
            if crop_height < 0:
                raise ValueError("crop_height must be positive")
            if inst_height <= crop_height:
                raise ValueError("inst_height must be greater than crop_height")
            d_h = 10 ** (0.979 * np.log10(max(crop_height, 1e-6)) - 0.154)
            zm_eff = inst_height - d_h
            z0_eff = crop_height * 0.123

        super().__init__(
            df=df,
            domain=domain,
            dx=dx,
            dy=dy,
            rs=rs,
            crop_height=crop_height,
            atm_bound_height=atm_bound_height,
            inst_height=inst_height,
            zm=zm,
            z0=z0,
            smooth_data=smooth_data,
            verbosity=verbosity,
            logger=logger,
        )

        # Store resolved values for internal use
        self.zm_eff = zm_eff
        self.z0_eff = z0_eff

        # Set up logger
        self.logger.setLevel(logging.DEBUG if self.verbosity > 1 else logging.WARNING)

        # Model constants
        self.k = 0.4  # von Karman constant
        self.oln = 5000.0  # neutral stability limit

        # Add RSL-specific parameters
        self.n_rsl = 2.75  # RSL height multiplier (2 ≤ n ≤ 5 per paper)
        self.rsl_params = {
            "conv": {"ps1": 0.8},  # p value for convective conditions
            "stab": {"ps1": 0.55},  # p value for stable conditions
        }

        # Validate input DataFrame
        self.df = self._validate_input_df(self.df)

        # Process input data
        self.prep_df_fields(
            zm=zm_eff,
            z0=z0_eff,
            atm_bound_height=self.atm_bound_height,
        )

        # Initialize basic attributes
        self.sigma_y = None
        self.sigma_y_correction = None
        self.fclim_2d = None
        self.f_2d = None

        self.domain = self._validate_domain(domain)
        self.dx = float(dx)
        self.dy = float(dy)
        self.nx = int(nx)
        self.ny = int(ny)
        self.rs = self._validate_rs(rs)

        self.smooth_data = bool(smooth_data)
        self.crop = bool(crop)
        self.rslayer = bool(rslayer)

        # Initialize model parameters (will be updated based on stability)
        self.initialize_model_parameters()

        # Set up computational domain
        self.define_domain()

        # Create xarray dataset
        self.create_xr_dataset()

        # Perform validity checks
        self.check_validity_ranges()

        # Handle stability regimes
        self.handle_stability_regimes()

    def _validate_input_df(self, df: pd.DataFrame) -> None:
        """
        Validate that the input DataFrame contains all required columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to validate.

        Raises
        ------
        ValueError
            If required columns are missing from the DataFrame.
        """
        df1 = df.copy()  # Make a copy to avoid modifying original

        # Rename fields to standard names
        df1 = df1.rename(
            columns={
                "V_SIGMA": "sigmav",
                "USTAR": "ustar",
                "wd": "wind_dir",
                "WD": "wind_dir",
                "MO_LENGTH": "ol",
                "ws": "umean",
                "WS": "umean",
            }
        )

        missing_cols = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in df1.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in input DataFrame: {missing_cols}"
            )

        # Check for invalid values
        for col in self.REQUIRED_COLUMNS:
            if df1[col].isnull().any():
                self.logger.warning(f"Found null values in column {col}")
            if not np.isfinite(df1[col]).all():
                self.logger.warning(f"Found non-finite values in column {col}")

        return df1

    def _validate_domain(self, domain: list) -> list:
        """
        Validate the domain boundaries.

        Parameters
        ----------
        domain : list of float
            A list with exactly four elements: [xmin, xmax, ymin, ymax].

        Returns
        -------
        list of float
            Validated domain list.

        Raises
        ------
        ValueError
            If the domain is not in the correct format.
        """
        if len(domain) != 4:
            raise ValueError("domain must be a list of [xmin, xmax, ymin, ymax]")
        domain = [float(x) for x in domain]
        if domain[0] >= domain[1] or domain[2] >= domain[3]:
            raise ValueError("Invalid domain bounds")
        return domain

    def _validate_rs(self, rs: list) -> list:
        """
        Validate the list of relative source area contributions.

        Parameters
        ----------
        rs : list of float
            List of values between 0 and 1 (exclusive).

        Returns
        -------
        list of float
            Sorted list of validated relative contributions.

        Raises
        ------
        ValueError
            If any value in the list is not in the (0, 1) range.
        """
        if not isinstance(rs, (list, np.ndarray)):
            raise ValueError("rs must be a list or array")
        rs = [float(r) for r in rs]
        if any(r <= 0 or r >= 1 for r in rs):
            raise ValueError("all rs values must be between 0 and 1")
        return sorted(rs)

    def _setup_logger(self) -> logging.Logger:
        """
        Set up a logger for the FFPModel class.

        Returns
        -------
        logging.Logger
            A configured logger instance.
        """
        logger = logging.getLogger("FFPModel")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def get_scalar_value(self, arr):
        """
        Convert an array-like object to a scalar float value.

        Parameters
        ----------
        arr : array-like or scalar
            The value to be converted.

        Returns
        -------
        float
            A scalar representation of the input.
        """
        if isinstance(arr, xr.DataArray):
            return float(arr.mean())
        elif isinstance(arr, np.ndarray):
            return float(np.mean(arr))
        else:
            return float(arr)

    def calc_crosswind_integrated_footprint(self, x_star):
        """
        Calculate the scaled crosswind-integrated footprint F̂y*(X̂*).

        Parameters
        ----------
        x_star : xarray.DataArray
            Scaled distance from the receptor.

        Returns
        -------
        xarray.DataArray
            The scaled crosswind-integrated footprint.
        """

        self.logger.debug("Calculating crosswind-integrated footprint...")

        try:
            # Parameters may be scalars OR time-varying DataArrays
            # Keep them as-is for broadcasting
            a = self.a
            b = self.b
            c = self.c
            d = self.d
            
            # If they're DataArrays, ensure they broadcast correctly with x_star
            # x_star shape: (time, x, y)
            # a, b, c, d might be: (time,) or scalar
            
            self.logger.debug(f"Parameter shapes: a={getattr(a, 'shape', 'scalar')}, "
                            f"b={getattr(b, 'shape', 'scalar')}, "
                            f"c={getattr(c, 'shape', 'scalar')}, "
                            f"d={getattr(d, 'shape', 'scalar')}")

            # Calculate x_star - d (broadcasting handles dimensions)
            x_star_modified = x_star - d
            
            # Mask for valid values
            mask = x_star_modified > 0.01

            # Calculate exponent with overflow protection
            # Use xr.where to handle the division safely
            safe_denominator = xr.where(
                x_star_modified > 0.01, 
                x_star_modified, 
                1.0
            )
            
            exp_arg = -c / safe_denominator
            exp_arg = xr.where(mask, exp_arg, -700)
            exp_arg = xr.where(exp_arg > -700, exp_arg, -700)
            
            # Calculate power term
            power_term = xr.where(
                mask,
                x_star_modified ** b,
                0.0
            )
            
            # Combine terms
            f_star = xr.where(
                mask,
                a * power_term * np.exp(exp_arg),
                0.0,
            )

            # Safety checks
            f_star = xr.where(np.isfinite(f_star), f_star, 0.0)
            f_star = xr.where(f_star < 1e10, f_star, 0.0)
            f_star = xr.where(f_star > 0, f_star, 0.0)

            self.logger.debug(
                f"f_star shape: {f_star.shape}, "
                f"range: {float(f_star.min()):.2e} to {float(f_star.max()):.2e}"
            )

            return f_star

        except Exception as e:
            self.logger.error(f"Error in crosswind integration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def get_source_area_contour(self, r, x_ru, x_rd, y_r):
        """
        Generate a contour dataset for a given source area.

        Parameters
        ----------
        r : float
            Relative contribution (0 < r < 1).
        x_ru : float
            Upwind distance.
        x_rd : float
            Downwind distance.
        y_r : float
            Crosswind extent.

        Returns
        -------
        xarray.Dataset
            Contour dataset with coordinates and footprint values.
        """
        # Get scalar values
        x_rd_val = self.get_scalar_value(x_rd)
        x_ru_val = self.get_scalar_value(x_ru)
        y_r_val = self.get_scalar_value(y_r)

        # Create distance arrays with unique, evenly spaced coordinates
        x = np.linspace(x_rd_val, x_ru_val, self.nx)
        y = np.linspace(-y_r_val, y_r_val, self.ny)

        # Ensure coordinates are unique by adding small offset where needed
        eps = np.finfo(float).eps  # Smallest possible float value
        x_unique = x + np.arange(len(x)) * eps
        y_unique = y + np.arange(len(y)) * eps

        # Create grid
        xx, yy = np.meshgrid(x_unique, y_unique)

        # The grid is already in the along-wind / crosswind frame:
        # xx = along-wind distance, yy = crosswind distance.
        # Pass xx directly so the parameterisations see the correct upwind
        # distance rather than the radial distance sqrt(xx²+yy²).
        x_star = self.calc_scaled_x(xx)
        sigma_y = self.calc_crosswind_spread(xx)
        f_y = self.calc_crosswind_integrated_footprint(x_star)

        # Calculate 2D footprint
        f_2d = (
            f_y / (np.sqrt(2 * np.pi) * sigma_y) * np.exp(-(yy**2) / (2.0 * sigma_y**2))
        )

        # Calculate contour level
        total_flux = np.sum(f_2d) * self.dx * self.dy
        sorted_f = np.sort(f_2d.flatten())[::-1]
        cumsum_f = np.cumsum(sorted_f) * self.dx * self.dy
        idx = np.searchsorted(cumsum_f / total_flux, r)
        if idx >= len(sorted_f):
            idx = len(sorted_f) - 1
        contour_level = sorted_f[idx]

        # Create output dataset with unique coordinates
        return xr.Dataset(
            {
                "contour_level": xr.DataArray(contour_level),
                "x": xr.DataArray(x_unique, dims=["x"]),
                "y": xr.DataArray(y_unique, dims=["y"]),
                "f": xr.DataArray(f_2d, dims=["y", "x"]),
            }
        )

    def calc_scaled_x(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the scaled distance X* based on wind and height parameters.

        Implements Eq. 7 of Kljun et al. (2015):

        .. math::

            X^* = \\frac{x}{z_m}\\left(1 - \\frac{z_m}{h}\\right)
                  \\left[\\ln\\frac{z_m}{z_0} - \\psi_M\\right]^{-1}

        i.e. ``X* = x/zm * (1 - zm/h) / Pi_4`` with
        ``Pi_4 = ln(zm/z0) - psi_M = u(zm)/(u* k)`` (Eqs. 6/7). This is the
        same scaled<->real conversion used in :meth:`calc_xr_footprint`, so the
        auxiliary source-area contour helpers stay consistent with the main
        climatology path.

        Parameters
        ----------
        x : float or ndarray
            Upwind distance from the receptor.

        Returns
        -------
        float or ndarray
            Scaled distance X*.
        """
        # Get mean values
        zm_mean = float(self.ds["zm"].mean())
        h_mean = float(self.ds["h"].mean())

        # Pi_4 = ln(zm/z0) - psi_M (Eq. 7); 1/Pi_4 maps real -> scaled distance.
        # calc_pi_4() returns the full Pi_4 (including the stability correction).
        pi4_mean = float(self.calc_pi_4().mean())

        # Calculate scaling
        result = x / zm_mean * (1 - zm_mean / h_mean) / pi4_mean

        return result

    def initialize_model_parameters(self):
        """
        Initialize core model parameters (a, b, c, d, ac, bc, cc).
        Ensures scalar values are assigned.
        """

        # Start with universal parameters as scalars
        self.a = 1.4524
        self.b = -1.9914
        self.c = 1.4622
        self.d = 0.1359

        # Crosswind dispersion parameters
        self.ac = 2.17
        self.bc = 1.66
        self.cc = 20.0
        
        self.logger.debug("Initialized with universal parameters")


    def check_rsl_validity(self):
        """
        Check if the measurement height is above the roughness sublayer.

        Returns
        -------
        pandas.Series
            Boolean mask of valid rows.
        """
        # Calculate RSL height
        h_rs = 10 * self.df["z0"]  # Roughness element height
        z_star = self.n_rsl * h_rs  # RSL height

        # Check if measurement height is above RSL
        rsl_valid = self.df["zm"] > z_star

        # Log RSL check results
        invalid_count = np.sum(~rsl_valid)
        if invalid_count > 0:
            self.logger.warning(
                f"{invalid_count} measurements within roughness sublayer "
                f"(zm <= {self.n_rsl:.1f} * h_rs)"
            )

        return rsl_valid

    def apply_rsl_corrections(self):
        """
        Apply roughness sublayer (RSL) corrections to crosswind dispersion.
        """
        # Calculate RSL parameters
        h_rs = 10 * self.ds["z0"]
        z_star = self.n_rsl * h_rs

        # Identify measurements within RSL
        in_rsl = self.ds["zm"] <= z_star

        if in_rsl.any():
            # Calculate scaling factors for RSL effects
            stability_param = self.ds["zm"] / self.ds["ol"]

            # Get appropriate ps1 value based on stability
            ps1 = xr.where(
                stability_param <= 0,
                self.rsl_params["conv"]["ps1"],
                self.rsl_params["stab"]["ps1"],
            )

            # Calculate RSL correction based on eq. C1-C3
            T = ps1 * self.ds["zm"] / self.ds["ustar"]
            sigma_y0 = self.ds["sigmav"] * T

            # Base crosswind spread as a function of x → expand over y to (x, y)
            sigma_y_1d = xr.DataArray(
                self.calc_crosswind_spread(self.x),
                dims=("x",),
                coords={"x": self.x},
            )
            sigma_y_grid = sigma_y_1d.expand_dims(y=self.y)  # (x, y)

            # Broadcast to (time, x, y)
            sigma_y_grid_3d = sigma_y_grid.expand_dims(time=self.ds.time)
            sigma_y0_3d = sigma_y0.broadcast_like(sigma_y_grid_3d)
            in_rsl_3d = in_rsl.broadcast_like(sigma_y_grid_3d)

            # Combine near-source and far-field terms where in RSL
            self.sigma_y = xr.where(
                in_rsl_3d,
                np.sqrt(sigma_y0_3d ** 2 + sigma_y_grid_3d ** 2),
                sigma_y_grid_3d,
            )

            # Blind-zone correction (time-only)
            self.x_min = self.ds["ustar"] * T

        else:
            # Outside RSL: keep base spread and shape it to (time, x, y)
            sigma_y_1d = xr.DataArray(
                self.calc_crosswind_spread(self.x),
                dims=("x",),
                coords={"x": self.x},
            )
            self.sigma_y = sigma_y_1d.expand_dims(y=self.y).expand_dims(time=self.ds.time)
            # No blind-zone correction needed; keep as NaN series
            self.x_min = xr.full_like(self.ds["ustar"], np.nan)


    def prep_df_fields(self, zm: float, z0: float, atm_bound_height: float):
        """
        Prepare and normalize input DataFrame fields for footprint calculation.

        Parameters
        ----------
        zm : float
            Effective measurement height above displacement height (m).
        z0 : float
            Aerodynamic roughness length (m).
        atm_bound_height : float
            Atmospheric boundary layer height (m).
        """
        self.df["zm"] = zm
        self.df["z0"] = z0
        self.df["h"] = atm_bound_height

        # Apply validity checks
        self._apply_validity_masks()

        # Drop invalid data
        self.df = self.df.dropna(subset=["sigmav", "wind_dir", "h", "ol"])
        self.ts_len = len(self.df)
        self.logger.debug(f"Valid input length: {self.ts_len}")

        # Add RSL validity check
        rsl_valid = self.check_rsl_validity()
        self.df["rsl_valid"] = rsl_valid

        # Additional RSL parameters
        self.df["h_rs"] = 10 * self.df["z0"]
        self.df["z_star"] = self.n_rsl * self.df["h_rs"]

        # Log RSL statistics
        mean_z_star = self.df["z_star"].mean()
        self.logger.info(
            f"Mean RSL height (z*): {mean_z_star:.1f}m, "
            f"Measurement height (zm): {zm}m"
        )

    def _apply_validity_masks(self):
        """
        Apply basic physical constraints to filter invalid data in the DataFrame.
        """
        self.df["zm"] = np.where(self.df["zm"] <= 0.0, np.nan, self.df["zm"])
        self.df["h"] = np.where(self.df["h"] <= 10.0, np.nan, self.df["h"])
        self.df["zm"] = np.where(self.df["zm"] > self.df["h"], np.nan, self.df["zm"])
        self.df["sigmav"] = np.where(self.df["sigmav"] < 0.0, np.nan, self.df["sigmav"])
        self.df["ustar"] = np.where(self.df["ustar"] <= 0.1, np.nan, self.df["ustar"])
        self.df["wind_dir"] = np.where(
            (self.df["wind_dir"] > 360.0) | (self.df["wind_dir"] < 0.0),
            np.nan,
            self.df["wind_dir"],
        )

    def define_domain(self):
        """
        Define the spatial computational domain and polar coordinate grids.
        """
        self.logger.info("Setting up computational domain...")

        try:
            # Create coordinate arrays
            if self.dx is None and self.nx is not None:
                self.x = np.linspace(self.domain[0], self.domain[1], self.nx + 1)
                self.y = np.linspace(self.domain[2], self.domain[3], self.ny + 1)
            else:
                self.x = np.arange(self.domain[0], self.domain[1] + self.dx, self.dx)
                self.y = np.arange(self.domain[2], self.domain[3] + self.dy, self.dy)

            self.logger.debug(f"Domain dimensions - x: {len(self.x)}, y: {len(self.y)}")

            # Create 2D grid with explicit dimensions
            self.xv, self.yv = np.meshgrid(self.x, self.y, indexing="ij")

            # Create polar coordinate grids as xarray DataArrays with explicit dimensions
            self.rho = xr.DataArray(
                np.sqrt(self.xv**2 + self.yv**2),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )

            self.theta = xr.DataArray(
                np.arctan2(self.xv, self.yv),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )

            # Initialize footprint grid with proper dimensions
            self.fclim_2d = xr.zeros_like(self.rho)

            self.logger.debug(
                f"Grid shapes - xv: {self.xv.shape}, rho: {self.rho.shape}"
            )
            self.logger.info("Domain setup completed successfully")

        except Exception as e:
            self.logger.error(f"Error in domain setup: {str(e)}")
            raise

    def create_xr_dataset(self):
        """Create xarray Dataset from input DataFrame."""
        self.df.index.name = "time"
        self.ds = xr.Dataset.from_dataframe(self.df)

    def handle_stability_regimes(self):
        """
        Classify stability regimes for diagnostics.

        Notes
        -----
        Kljun et al. (2015) deliberately use a single, stability-independent
        parameter set (a, b, c, d) for the scaled crosswind-integrated
        footprint (Appendix A: a=1.4524, b=-1.9914, c=1.4622, d=0.1359). The
        stability dependence enters only through the scaled distance X* and the
        crosswind-spread scaling. Therefore this method no longer overrides the
        universal coefficients; it only classifies regimes and records a
        diagnostic flag.

        Returns
        -------
        xarray.Dataset
            Dataset with regime-specific masks.
        """

        # Calculate stability parameter
        stability_param = self.ds["zm"] / self.ds["ol"]

        # Define regime boundaries
        regimes = xr.Dataset()
        regimes["strongly_unstable"] = stability_param <= -15.5
        regimes["unstable"] = (stability_param > -15.5) & (stability_param < -0.1)
        regimes["neutral"] = (stability_param >= -0.1) & (stability_param <= 0.1)
        regimes["stable"] = (stability_param > 0.1) & (
            stability_param < self.oln / self.ds["zm"]
        )
        regimes["strongly_stable"] = stability_param >= self.oln / self.ds["zm"]

        # Log regime statistics
        for regime in regimes:
            count = int(regimes[regime].sum())
            percentage = (count / self.ts_len) * 100
            self.logger.info(f"{regime}: {count} points ({percentage:.1f}%)")

        # Add stability flags to dataset
        self.ds["stability_regime"] = xr.where(
            regimes["strongly_unstable"], -2,
            xr.where(regimes["unstable"], -1,
            xr.where(regimes["neutral"], 0,
            xr.where(regimes["stable"], 1, 2)))
        )

        return regimes

    def check_validity_ranges(self):
        """
        Check validity ranges according to equation 27 and other constraints from Kljun et al. (2015).

        Implements the following validity checks:
        1. Height validity (Eq. 27): 20z₀ < zm < he
           where he ≈ 0.8h is the entrainment height
        2. Stability validity (Eq. 27): -15.5 ≤ zm/L
        3. Turbulence validity: u* > 0.1 m/s, σv > 0
        4. Roughness sublayer validity: zm > z* ≈ n*hrs
           where hrs ≈ 10z₀ and 2 ≤ n ≤ 5 (Section 2)

        Returns
        -------
        xr.Dataset: Dataset containing per-timestep boolean masks for:
        - height_valid: True where 20z₀ < zm < he
        - stability_valid: True where -15.5 ≤ zm/L
        - turbulence_valid: True where u* > 0.1 and σv > 0
        - rsl_valid: True where measurement height above RSL
        Combined per-timestep validity is stored in self.valid_footprint (dims: time).
        """
        # Per-timestep boolean validity flags (dims: time)
        height_valid = (
            (self.ds["zm"] > 20 * self.ds["z0"]) & (self.ds["zm"] < 0.8 * self.ds["h"])
        )
        stability_valid = self.ds["zm"] / self.ds["ol"] >= -15.5
        turbulence_valid = (self.ds["ustar"] > 0.1) & (self.ds["sigmav"] > 0)

        # RSL validity was pre-computed in prep_df_fields() and stored in self.ds
        rsl_valid = self.ds["rsl_valid"].astype(bool)

        # Combined per-timestep validity mask (dims: time)
        self.valid_footprint = height_valid & stability_valid & turbulence_valid & rsl_valid

        n_invalid = int((~self.valid_footprint).sum())
        if n_invalid > 0:
            self.logger.warning(
                f"{n_invalid}/{self.ts_len} timesteps excluded: outside "
                f"Kljun et al. (2015) validity bounds (Eq. 27)"
            )

        validity_mask = xr.Dataset({
            "height_valid": height_valid,
            "stability_valid": stability_valid,
            "turbulence_valid": turbulence_valid,
            "rsl_valid": rsl_valid,
        })
        return validity_mask

    def calc_pi_4(self):
        """
        Calculate the dimensionless wind-speed group
        Π4 = u(zm)/(u* k) = ln(zm/z0) - ψM, following Kljun et al. (2015).

        Π4 is the scaled<->real conversion factor for the along-wind distance
        (Eq. 6/7) and must be positive. Callers use the return value of this
        method *directly* as Π4 (do **not** subtract it from ``ln(zm/z0)``
        again).

        Returns
        -------
        xr.DataArray
            Π4 = ln(zm/z0) - ψM. In neutral conditions ψM = 0 so
            Π4 = ln(zm/z0).
        """
        stability_param = self.ds["zm"] / self.ds["ol"]

        # Initialize psi_m array
        psi_m = xr.zeros_like(stability_param)

        # Determine stability regimes.
        # Stable: 0 < L < oln. Unstable: L < 0.
        # Near-neutral (|L| >= oln) keeps psi_m = 0.
        stable_mask = (self.ds["ol"] > 0) & (self.ds["ol"] < self.oln)
        unstable_mask = self.ds["ol"] < 0
        neutral_mask = ~(stable_mask | unstable_mask)

        # Calculate psi_m for each stability regime
        # Stable conditions
        psi_m = xr.where(stable_mask, -5.3 * stability_param, psi_m)

        # Unstable conditions
        psi_m = xr.where(unstable_mask, self.calc_unstable_psi(stability_param), psi_m)

        # Neutral conditions - psi_m remains zero
        self.logger.debug(
            f"Stability regime counts - Stable: {stable_mask.sum().values}, "
            f"Unstable: {unstable_mask.sum().values}, "
            f"Neutral: {neutral_mask.sum().values}"
        )

        return np.log(self.ds["zm"] / self.ds["z0"]) - psi_m

    def calc_unstable_psi(self, stability_param):
        """
        Calculate ψM for unstable atmospheric conditions.

        Parameters
        ----------
        stability_param : xarray.DataArray
            Stability parameter zm/L.

        Returns
        -------
        xarray.DataArray
            Stability correction ψM.
        """
        # Limit stability parameter to prevent numerical issues
        stability_param = xr.where(stability_param < -15.5, -15.5, stability_param)

        chi = (1 - 19 * stability_param) ** 0.25

        # Implement the full equation with better numerical handling
        return (
            np.log((1 + chi**2) / 2)
            + 2 * np.log((1 + chi) / 2)
            - 2 * np.arctan(chi)
            + np.pi / 2
        )

    def apply_paper_smoothing(self):
        """
        Apply 3x3 smoothing from Kljun et al. paper with mass conservation.
        """
        self.logger.debug("Starting paper smoothing...")

        try:
            # Define smoothing kernel from paper
            skernel = np.array([
                [0.05, 0.1, 0.05],
                [0.1, 0.4, 0.1],
                [0.05, 0.1, 0.05]
            ])

            # Store original for validation
            original = self.fclim_2d.copy()
            original_sum = float(original.sum())
            
            self.logger.debug(
                f"Pre-smoothing stats - Min: {float(original.min()):.2e}, "
                f"Max: {float(original.max()):.2e}, Sum: {original_sum:.2e}"
            )

            # Apply convolution twice as specified in paper
            smoothed = original.copy()
            for i in range(2):
                smoothed_values = signal.convolve2d(
                    smoothed.values,
                    skernel,
                    mode='same',
                    boundary='fill',  # Zero-pad edges; 'wrap' would bleed opposite-edge mass
                    fillvalue=0
                )
                smoothed = xr.DataArray(
                    smoothed_values,
                    dims=smoothed.dims,
                    coords=smoothed.coords
                )
                self.logger.debug(f"Smoothing iteration {i + 1} complete")

            # CRITICAL: Preserve total mass
            smoothed_sum = float(smoothed.sum())
            if smoothed_sum > 1e-10:
                smoothed = smoothed * (original_sum / smoothed_sum)
                self.logger.debug(
                    f"Mass conservation: {original_sum:.6e} -> {float(smoothed.sum()):.6e}"
                )

            # Validation
            if np.isnan(smoothed).any() or not np.any(smoothed):
                self.logger.warning(
                    "Smoothing produced invalid results, reverting to original"
                )
                return original

            self.logger.debug(
                f"Post-smoothing stats - Min: {float(smoothed.min()):.2e}, "
                f"Max: {float(smoothed.max()):.2e}"
            )

            return smoothed

        except Exception as e:
            self.logger.error(f"Error in smoothing: {str(e)}")
            self.logger.debug("Returning original unsmoothed data")
            return self.fclim_2d

    def verify_results(self, data, name="data"):
        """
        Check for NaN or infinite values in the given data.

        Parameters
        ----------
        data : array-like or xarray.DataArray
            The data to verify.
        name : str, optional
            Descriptive name for logging.
        """
        try:
            if isinstance(data, xr.DataArray):
                if data.isnull().any():
                    self.logger.warning(f"NaN values detected in {name}")
                if not xr.DataArray.all(data.where(data.notnull())):
                    self.logger.warning(f"Non-finite values detected in {name}")

            elif isinstance(data, np.ndarray):
                if np.any(np.isnan(data)):
                    self.logger.warning(f"NaN values detected in {name}")
                if not np.all(np.isfinite(data)):
                    self.logger.warning(f"Non-finite values detected in {name}")

            else:
                # Try to convert to numpy array first
                try:
                    arr = np.asarray(data)
                    if np.any(np.isnan(arr)):
                        self.logger.warning(f"NaN values detected in {name}")
                    if not np.all(np.isfinite(arr)):
                        self.logger.warning(f"Non-finite values detected in {name}")
                except:
                    self.logger.warning(
                        f"Could not verify {name} for NaN/infinite values"
                    )

        except Exception as e:
            self.logger.warning(f"Error verifying {name}: {str(e)}")

    def verify_data(self, data: xr.DataArray, name: str) -> bool:
        """
        Verify the content and structure of a DataArray.

        Parameters
        ----------
        data : xarray.DataArray
            Data array to check.
        name : str
            Variable name for logging.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        if data is None:
            self.logger.error(f"{name} is None")
            return False

        if not isinstance(data, xr.DataArray):
            self.logger.error(f"{name} is not a DataArray, got {type(data)}")
            return False

        if data.size == 0:
            self.logger.error(f"{name} is empty")
            return False

        if np.all(np.isnan(data)):
            self.logger.error(f"All values in {name} are NaN")
            return False

        # Log statistics for debugging
        self.logger.debug(f"{name} statistics:")
        self.logger.debug(f"  Shape: {data.shape}")
        self.logger.debug(f"  NaN count: {np.sum(np.isnan(data))}")
        self.logger.debug(f"  Min: {float(data.min())}")
        self.logger.debug(f"  Max: {float(data.max())}")
        self.logger.debug(f"  Mean: {float(data.mean())}")

        return True

    def calc_xr_footprint(self):
        """
        Compute the flux footprint using xarray operations.

        Returns
        -------
        xarray.DataArray
            Climatological flux footprint.
        """
        self.logger.info("Starting footprint calculation...")


        try:
            # Step 1: Ensure spatial grids have correct dims
            if not hasattr(self, 'rho') or self.rho is None:
                self.logger.error("Domain not initialized")
                raise ValueError("Call define_domain() first")

            self.logger.debug(f"rho dims: {self.rho.dims}, shape: {self.rho.shape}")
            self.logger.debug(f"theta dims: {self.theta.dims}, shape: {self.theta.shape}")
            
            # Step 2: Calculate rotated theta - EXPLICIT dimension order
            # theta: (x, y), wind_dir: (time) -> result should be (time, x, y)
            wind_dir_rad = self.ds["wind_dir"] * np.pi / 180.0
            
            # Expand spatial grids to include time dimension FIRST
            rho_3d = self.rho.expand_dims(time=self.ds.time, axis=0)
            theta_3d = self.theta.expand_dims(time=self.ds.time, axis=0)
            
            # Now broadcast works correctly: (time, x, y) - (time,) -> (time, x, y)
            self.rotated_theta = theta_3d - wind_dir_rad

            self.logger.debug(f"rotated_theta dims: {self.rotated_theta.dims}, shape: {self.rotated_theta.shape}")

            # Pre-compute RSL corrections (sigma_y, x_min) when rslayer is enabled.
            # Must happen after rotated_theta is set; calc_scaled_distance_rsl() uses it.
            if self.rslayer:
                self.apply_rsl_corrections()

            # Step 3: Dimensionless wind-speed group Pi_4 = ln(zm/z0) - psi_M
            # (Eq. 6/7). This is the scaled->real conversion factor and must be
            # > 0. calc_pi_4() already returns the full Pi_4.
            pi4 = self.calc_pi_4()
            self.logger.debug(f"pi4 dims: {pi4.dims}, shape: {pi4.shape}")
            valid_profile = pi4 > 0

            # Step 4: Calculate scaled distance - all should be (time, x, y)
            if self.rslayer:
                # RSL-corrected path: blind-zone adjustment via self.x_min (set above)
                xstar_ci = xr.where(valid_profile, self.calc_scaled_distance_rsl(), 0.0)
            else:
                xstar_ci = xr.where(
                    valid_profile,
                    rho_3d * np.cos(self.rotated_theta)
                    / self.ds["zm"]
                    * (1.0 - self.ds["zm"] / self.ds["h"])
                    / pi4,
                    0.0,
                )

            self.logger.debug(f"xstar_ci dims: {xstar_ci.dims}, shape: {xstar_ci.shape}")
            self.logger.debug(
                f"xstar_ci range: {float(xstar_ci.min()):.2e} to {float(xstar_ci.max()):.2e}"
            )

            # Step 5: Calculate footprint components.
            # calc_crosswind_integrated_footprint returns the *scaled* Fy_hat*;
            # convert it to the real-scale crosswind-integrated footprint Fy
            # via the Jacobian (1/zm)(1 - zm/h)/Pi_4 (Eq. 6/7). Without this
            # factor the 2D footprint carries the wrong units and does not
            # integrate to unity.
            f_ci_star = self.calc_crosswind_integrated_footprint(xstar_ci)
            f_ci = f_ci_star / self.ds["zm"] * (1.0 - self.ds["zm"] / self.ds["h"]) / pi4
            self.logger.debug(f"f_ci dims: {f_ci.dims}, shape: {f_ci.shape}")
            self._log_array_stats("crosswind_integrated_footprint", f_ci)

            if self.rslayer and self.sigma_y is not None:
                # Use RSL-corrected crosswind spread from apply_rsl_corrections()
                sigy = self.sigma_y
            else:
                sigy = self.calc_crosswind_spread(xstar_ci)
            self.logger.debug(f"sigy dims: {sigy.dims}, shape: {sigy.shape}")
            self._log_array_stats("crosswind_dispersion", sigy)

            # Step 6: Calculate 2D footprint - all inputs are (time, x, y)
            self.f_2d = xr.where(
                sigy > 0,
                f_ci / (np.sqrt(2 * np.pi) * sigy)
                * np.exp(
                    -((rho_3d * np.sin(self.rotated_theta))**2) / (2.0 * sigy**2)
                ),
                0.0,
            )
            
            self.logger.debug(f"f_2d dims: {self.f_2d.dims}, shape: {self.f_2d.shape}")
            self._log_array_stats("2d_footprint", self.f_2d)

            # Zero out timesteps outside the Kljun et al. (2015) validity bounds
            # (Eq. 27) so they do not contribute weight to the climatology.
            self.f_2d = xr.where(self.valid_footprint, self.f_2d, 0.0)

            # Step 7: Calculate climatology - sum over TIME dimension
            footprint_sum = self.f_2d.sum(dim="time")
            self.logger.debug(f"footprint_sum dims: {footprint_sum.dims}, shape: {footprint_sum.shape}")

            # Count only timesteps that produced a non-zero footprint.
            # Invalid timesteps (sigy <= 0) are zeroed out and must not count
            # in the denominator, otherwise the climatology is underestimated.
            valid_count = int((self.f_2d.sum(dim=("x", "y")) > 0).sum())
            self.logger.debug(f"Valid timesteps: {valid_count} / {self.ts_len}")

            total_sum = float(footprint_sum.sum())
            self.logger.debug(f"Total sum before normalization: {total_sum:.2e}")

            if total_sum < 1e-10:
                self.logger.warning("Near-zero sum, using uniform distribution")
                self.fclim_2d = xr.ones_like(self.rho) / (len(self.x) * len(self.y))
            else:
                # Normalize: divide by number of valid timesteps only
                self.fclim_2d = footprint_sum / max(valid_count, 1)
                
                # CRITICAL: Verify normalization
                integrated_flux = float(self.fclim_2d.sum() * self.dx * self.dy)
                self.logger.info(f"Integrated flux: {integrated_flux:.3f} (should be ~1.0)")
                
                if integrated_flux < 0.8:
                    self.logger.warning(
                        f"Only captured {integrated_flux*100:.1f}% of flux - "
                        f"consider increasing domain size"
                    )

            self._log_array_stats("climatology", self.fclim_2d)

            # Step 8: Apply smoothing
            if self.smooth_data:
                self.fclim_2d = self.apply_paper_smoothing()

            self.logger.info("Footprint calculation completed successfully")
            return self.fclim_2d

        except Exception as e:
            self.logger.error(f"Error in footprint calculation: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise

    def normalize_footprint(self):
        """
        Ensure footprint integrates to 1.0 after calculation.
        """
        current_flux = float(self.fclim_2d.sum() * self.dx * self.dy)
        
        if current_flux > 1e-10:
            self.logger.info(f"Normalizing footprint: {current_flux:.3f} -> 1.000")
            self.fclim_2d = self.fclim_2d / current_flux
            
            # Verify
            new_flux = float(self.fclim_2d.sum() * self.dx * self.dy)
            self.logger.info(f"After normalization: {new_flux:.6f}")
        else:
            self.logger.error("Cannot normalize: footprint sum is near zero")

    def _log_array_stats(self, name, arr):
        """
        Log statistics for an array.

        Parameters
        ----------
        name : str
            Descriptive name.
        arr : xarray.DataArray
            The array to summarize.
        """
        self.logger.debug(f"{name} statistics:")
        self.logger.debug(f"  Shape: {arr.shape}")
        self.logger.debug(f"  NaN count: {xr.where(np.isnan(arr), 1, 0).sum()}")
        self.logger.debug(f"  Min: {float(arr.min())}")
        self.logger.debug(f"  Max: {float(arr.max())}")
        self.logger.debug(f"  Mean: {float(arr.mean())}")

    def calc_scaled_distance_rsl(self):
        """
        Calculate scaled distance X* with roughness sublayer correction.

        Returns
        -------
        xarray.DataArray
            Corrected scaled distance.
        """

        # Basic scaled distance calculation (calc_pi_4 returns Pi_4 directly)
        xstar_base = (
            self.rho
            * np.cos(self.rotated_theta)
            / self.ds["zm"]
            * (1.0 - self.ds["zm"] / self.ds["h"])
            / self.calc_pi_4()
        )

        # Apply RSL correction where needed
        in_rsl = self.ds["zm"] <= self.ds["z_star"]

        if in_rsl.any():
            # Calculate blind zone adjustment
            x_adj = xr.where(in_rsl, self.x_min * np.cos(self.rotated_theta), 0.0)

            # Adjust scaled distance
            xstar_adj = xr.where(in_rsl, xstar_base + x_adj / self.ds["zm"], xstar_base)

            return xstar_adj

        return xstar_base

    def calculate_pi_groups(self):
        """
        Calculate dimensionless Π groups for analysis.

        Returns
        -------
        tuple of xarray.DataArray
            Π₁, Π₂, Π₃, Π₄ values.
        """
        if not hasattr(self, "f_2d"):
            raise AttributeError(
                "f_2d must be initialized before calculating pi groups"
            )

        # Π1 = fy*zm
        pi_1 = self.f_2d * self.ds["zm"]

        # Π2 = x/zm
        pi_2 = self.rho * np.cos(self.rotated_theta) / self.ds["zm"]

        # Π3 = (h - zm)/h = 1 - zm/h
        pi_3 = 1 - self.ds["zm"] / self.ds["h"]

        # Π4 = u(zm)/(u*k) = ln(zm/z0) - ψM   (calc_pi_4 already returns Π4)
        pi_4 = self.calc_pi_4()

        return pi_1, pi_2, pi_3, pi_4

    def calc_2d_footprint(self, f_ci_dummy, sigy_dummy):
        """
        Calculate the two-dimensional footprint distribution f(x,y).

        Implementation of Equation 10 from Kljun et al. (2015):
        f(x,y) = fy(x) * (1/(sqrt(2π)σy)) * exp(-y^2/(2σy^2))

        Where:
        - fy(x) is the crosswind-integrated footprint
        - σy is the standard deviation of crosswind spread
        - y is the crosswind distance from the centerline

        Args:
            f_ci_dummy: Crosswind-integrated footprint
            sigy_dummy: Scaled crosswind dispersion

        Returns:
            xr.DataArray: Two-dimensional footprint [m^-2]
        """
        return xr.where(
            sigy_dummy > 0,
            f_ci_dummy
            / (np.sqrt(2 * np.pi) * sigy_dummy)
            * np.exp(
                -((self.rho * np.sin(self.rotated_theta)) ** 2) / (2.0 * sigy_dummy**2)
            ),
            0.0,
        )

    def calculate_source_areas(self):
        """
        Compute source areas for all specified relative contributions.

        Returns
        -------
        dict
            Dictionary of contour datasets by contribution level.
        """
        source_areas = {}

        # Calculate for each requested relative contribution
        for r in self.rs:
            if not 0.1 <= r <= 0.9:
                self.logger.warning(f"Skipping r={r}, outside valid range 0.1-0.9")
                continue

            # Calculate extents
            x_r = self.calc_crosswind_integrated_extent(r)
            x_ru, x_rd = self.calc_peak_based_limits(r)
            y_r = self.calc_crosswind_extent(r, x_ru, x_rd)

            # Store results as xarray objects
            source_areas[f"r_{int(r * 100)}"] = {
                "x_r": xr.DataArray(x_r, name="extent"),
                "x_ru": xr.DataArray(x_ru, name="upwind_extent"),
                "x_rd": xr.DataArray(x_rd, name="downwind_extent"),
                "y_r": xr.DataArray(y_r, name="crosswind_extent"),
                "contour": self.get_source_area_contour(r, x_ru, x_rd, y_r),
            }

        return source_areas

    def calc_crosswind_integrated_extent(self, r: float) -> xr.DataArray:
        """
        Calculate crosswind-integrated footprint extent for relative contribution r.
        Following Equations 23-25 of the paper.

        Args:
            r: Relative contribution (between 0.1 and 0.9)

        Returns:
            xr.DataArray: Distance from receptor containing fraction r of footprint
        """
        # Calculate scaled extent from Eq. 24
        x_star_r = -self.c / np.log(r) + self.d

        # Convert to real scale using Eq. 25 (calc_pi_4 returns Pi_4 directly)
        x_r = (
            x_star_r
            * self.ds["zm"]
            * (1 - self.ds["zm"] / self.ds["h"]) ** -1
            * self.calc_pi_4()
        )

        return x_r
    

    def calc_peak_based_limits(self, r: float) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Calculate upwind and downwind distances from the peak location for fraction r.
        Following Eq. 26 of the paper.

        Args:
            r: Relative contribution (between 0.1 and 0.9)

        Returns:
            Tuple containing upwind and downwind distances from peak
        """
        # Calculate peak location (Eq. 20)
        x_star_max = -self.c / self.b + self.d

        # Get x_star_r from crosswind integration
        x_star_r = -self.c / np.log(r) + self.d

        # Calculate downwind limit (Eq. 26 with first set of parameters)
        x_star_rd = 0.44 * x_star_r**-0.77 + 0.24

        # New fixed condition:
        condition = (x_star_max < x_star_r) & (x_star_r <= 1.5)

        # Calculate upwind limit with split conditions (Eq. 26)
        x_star_ru = xr.where(
            condition,
            0.60 * x_star_r**1.32 + 0.61,
            0.96 * x_star_r**1.01 + 0.19,
        )

        # Convert to real scale
        x_rd = self.scale_to_real_distance(x_star_rd)
        x_ru = self.scale_to_real_distance(x_star_ru)

        return x_ru, x_rd

    def calc_real_footprint_peak(
        self,
        zm: Union[float, xr.DataArray],
        h: Union[float, xr.DataArray],
        u_zm: Union[float, xr.DataArray] = None,
        ustar: Union[float, xr.DataArray] = None,
        z0: Union[float, xr.DataArray] = None,
        k: float = 0.4,
    ) -> Union[float, xr.DataArray]:
        """
        Calculate the real-scale footprint peak location.
        Based on Equations 20-22 of Kljun et al. (2015).
        Works with both scalar values and xarray DataArrays.

        Args:
            zm: Measurement height [m]
            h: Boundary layer height [m]
            u_zm: Mean wind speed at measurement height [m/s], optional
            ustar: Friction velocity [m/s], optional
            z0: Roughness length [m], optional
            k: von Karman constant (default=0.4)

        Returns:
            Real distance to footprint peak [m]
            Returns DataArray if inputs are DataArrays, float if inputs are scalars
        """
        # Calculate scaled peak location (Eq. 20)
        x_star_max = -self.c / self.b + self.d  # Should evaluate to 0.87

        # Check which conversion method to use based on available inputs
        if u_zm is not None and ustar is not None:
            # Use Eq. 21 with wind speed and friction velocity
            x_max = x_star_max * zm * (1 - zm / h) ** -1 * u_zm / (ustar * k)

        elif z0 is not None:
            # Use Eq. 22 with roughness length and stability correction.
            # calc_pi_4() already returns Pi_4 = ln(zm/z0) - psi_M.
            pi4 = self.calc_pi_4()
            x_max = x_star_max * zm * (1 - zm / h) ** -1 * pi4

        else:
            raise ValueError("Must provide either (u_zm, ustar) or z0")

        # Log output differently for scalar vs DataArray
        if isinstance(x_max, xr.DataArray):
            self.logger.debug(
                f"Calculated footprint peak with mean at x = {float(x_max.mean()):.2f} m"
            )
        else:
            self.logger.debug(f"Calculated footprint peak at x = {x_max:.2f} m")

        return x_max

    def calc_scaled_footprint_peak(self) -> float:
        """
        Calculate the scaled footprint peak location (X*max).
        Based on Equation 20 of Kljun et al. (2015).

        Returns:
            float: Scaled distance to footprint peak (constant)
        """
        return -self.c / self.b + self.d  # Should evaluate to 0.87

    def scale_to_real_distance(self, x_star: xr.DataArray) -> xr.DataArray:
        """
        Convert scaled distance to real distance using Eq. 6/7.

        Args:
            x_star: Scaled distance X*

        Returns:
            xr.DataArray: Real-scale distance
        """
        return (
            x_star
            * self.ds["zm"]
            * (1 - self.ds["zm"] / self.ds["h"]) ** -1
            * self.calc_pi_4()  # calc_pi_4 returns Pi_4 directly
        )

    def calc_crosswind_extent(
        self, r: float, x_ru: xr.DataArray, x_rd: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate crosswind extent of source area using parameterized relationships.

        Args:
            r: Relative contribution
            x_ru: Upwind distance from peak
            x_rd: Downwind distance from peak

        Returns:
            xr.DataArray: Crosswind extent at each distance
        """
        # Calculate sigma_y at peak location using per-timestep parameters
        x_peak = self.calc_real_footprint_peak(
            self.ds["zm"], self.ds["h"], self.ds["umean"], self.ds["ustar"]
        )
        pi4 = self.calc_pi_4()
        x_star_peak = (
            x_peak / self.ds["zm"] * (1.0 - self.ds["zm"] / self.ds["h"]) / pi4
        )
        sigma_y_peak = self.calc_crosswind_spread(x_star_peak)

        # Scale crosswind extent based on distance from peak
        def scale_sigma(x_dist):
            return sigma_y_peak * np.sqrt(x_dist / x_peak)

        y_r = xr.where(x_rd <= x_peak, scale_sigma(x_rd), scale_sigma(x_ru))

        return y_r

    def calc_crosswind_spread(
        self, x: Union[float, np.ndarray, "xr.DataArray"]
    ) -> Union[float, np.ndarray, "xr.DataArray"]:
        r"""
        Calculate the standard deviation of cross-wind spread :math:`\sigma_y`.

        Implements Eqs. 18-19 from *Kljun et al., 2015*:

        .. math::

            \sigma_y^* = a_c \sqrt{\frac{b_c \lvert X^* \rvert^2}
                                        {1 + c_c \lvert X^* \rvert}}

            \sigma_y   = \frac{\sigma_y^*}{\text{scale\_const}}
                        \; z_m \; \sigma_v / u_*

        where :math:`a_c = 2.17`, :math:`b_c = 1.66`, :math:`c_c = 20.0`, and
        ``scale_const`` depends on stability (see paper).

        The method dispatches on the type of *x*:

        * **xarray DataArray** – *x* is treated as the pre-scaled dimensionless
          distance :math:`X^*`.  Per-timestep stability parameters from
          ``self.ds`` are used, and an ``xr.DataArray`` is returned.  Use this
          path for climatology calculations where time-varying parameters matter.
        * **numpy array / float** – *x* is treated as the raw upwind distance
          [m].  ``calc_scaled_x`` converts it to :math:`X^*` using time-mean
          parameters, and a numpy array / float is returned.  Use this path for
          single-footprint visualisation where a representative spread is needed.

        Parameters
        ----------
        x : float, ndarray, or xr.DataArray
            For numpy/float: raw upwind distance [m].
            For DataArray: pre-scaled dimensionless distance :math:`X^*`.

        Returns
        -------
        float, ndarray, or xr.DataArray
            Cross-wind spread :math:`\sigma_y` [m].
        """
        if isinstance(x, xr.DataArray):
            # xarray path: x is pre-scaled X*, use per-timestep parameters
            try:
                sigma_y_star = self.ac * np.sqrt(
                    (self.bc * np.abs(x) ** 2) / (1 + self.cc * np.abs(x))
                )
                stability = self.ds["zm"] / self.ds["ol"]
                scale_const = xr.where(
                    self.ds["ol"] <= 0,
                    1e-5 * np.maximum(np.abs(stability), 1e-10) ** (-1) + 0.80,
                    1e-5 * np.maximum(np.abs(stability), 1e-10) ** (-1) + 0.55,
                )
                scale_const = xr.where(scale_const > 1.0, 1.0, scale_const)
                return (
                    sigma_y_star
                    / scale_const
                    * self.ds["zm"]
                    * self.ds["sigmav"]
                    / self.ds["ustar"]
                )
            except Exception as e:
                self.logger.error(f"Error calculating crosswind spread: {str(e)}")
                raise
        else:
            # Numpy path: x is raw upwind distance, scale with time-mean parameters
            zm_mean = float(self.ds["zm"].mean())
            ol_mean = float(self.ds["ol"].mean())
            ustar_mean = float(self.ds["ustar"].mean())
            sigmav_mean = float(self.ds["sigmav"].mean())

            x_star = self.calc_scaled_x(x)
            # Use |X*| so points behind the receptor don't drive the denominator negative
            x_star_abs = np.abs(x_star)
            sigma_y_star = self.ac * np.sqrt(
                (self.bc * x_star_abs**2) / (1 + self.cc * x_star_abs)
            )

            stability_param_mean = zm_mean / ol_mean
            if stability_param_mean <= 0:
                scale_const = 1e-5 * abs(stability_param_mean) ** (-1) + 0.80
            else:
                scale_const = 1e-5 * abs(stability_param_mean) ** (-1) + 0.55
            scale_const = min(scale_const, 1.0)

            return sigma_y_star / scale_const * zm_mean * sigmav_mean / ustar_mean

    def plot_footprint(self, config=None, ax=None, show_contours=True, levels=10):
        """
        Plot the footprint climatology with optional contours.

        Args:
            config: Configuration object containing metadata (optional)
            ax: Matplotlib axis to plot on (optional)
            show_contours: Whether to show contour lines
            levels: Number of contour levels or list of levels

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 8))
            else:
                fig = ax.figure

            # Create meshgrid for plotting if needed
            if not hasattr(self, "X") or not hasattr(self, "Y"):
                self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

            self.logger.debug(f"Domain shapes - x: {len(self.x)}, y: {len(self.y)}")
            self.logger.debug(f"Footprint climatology shape: {self.fclim_2d.shape}")
            self.logger.debug(
                f"Footprint climatology stats - min: {float(self.fclim_2d.min())}, max: {float(self.fclim_2d.max())}"
            )

            # Plot filled contours
            contourf = ax.contourf(
                self.X, self.Y, self.fclim_2d, levels=levels, cmap="YlOrRd"
            )
            plt.colorbar(contourf, ax=ax, label="Flux footprint")

            # Add contour lines if requested
            if show_contours:
                contour = ax.contour(
                    self.X,
                    self.Y,
                    self.fclim_2d,
                    levels=levels,
                    colors="k",
                    alpha=0.5,
                    linewidths=0.5,
                )
                ax.clabel(contour, inline=True, fontsize=8, fmt="%.1e")

            # Add tower location
            ax.plot(0, 0, "k^", markersize=10, label="Tower")

            # Set labels and title
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("Distance [m]")

            # Try to get site name from config if available
            title = "Flux Footprint"
            if config is not None:
                try:
                    site_name = config.get("METADATA", "site_name", fallback="").strip()
                    if site_name:
                        title = f"Flux Footprint - {site_name}"
                except Exception as e:
                    self.logger.warning(
                        f"Could not get site name from config: {str(e)}"
                    )

            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            ax.legend()

            return fig, ax

        except Exception as e:
            self.logger.error(f"Error plotting footprint: {str(e)}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise

    def run(self, return_result: bool = True) -> Optional[xr.Dataset]:
        """
        Execute the FFP model calculations.

        Args:
            return_result: If True, returns complete results dataset

        Returns:
            Optional[xr.Dataset]: Dataset containing footprint results
        """
        self.logger.info("Starting FFP model calculations...")

        try:
            # Ensure domain is properly initialized
            if not hasattr(self, "x") or not hasattr(self, "y"):
                self.define_domain()

            # Initialize fclim_2d with zeros before calculations
            self.fclim_2d = xr.DataArray(
                np.zeros((len(self.x), len(self.y))),
                dims=("x", "y"),
                coords={"x": self.x, "y": self.y},
            )

            # Perform main calculations
            self.fclim_2d = self.calc_xr_footprint()  # Now returns fclim_2d directly

            # Rescale so the climatology integrates to ~1 over the domain
            self.normalize_footprint()

            # Add validation checks
            if self.fclim_2d is None:
                self.logger.error("fclim_2d is None after calculations")
                raise ValueError("Footprint climatology calculation failed")

            if not isinstance(self.fclim_2d, xr.DataArray):
                self.logger.error(
                    f"fclim_2d is not a DataArray, got {type(self.fclim_2d)}"
                )
                raise ValueError("Invalid footprint climatology type")

            if np.all(np.isnan(self.fclim_2d)):
                self.logger.error("All values in fclim_2d are NaN")
                raise ValueError("Invalid footprint climatology values")

            if return_result:
                self.logger.debug(f"Domain shapes - x: {len(self.x)}, y: {len(self.y)}")
                self.logger.debug(f"Footprint climatology shape: {self.fclim_2d.shape}")
                self.logger.debug(
                    f"Footprint climatology stats - min: {float(self.fclim_2d.min())}, max: {float(self.fclim_2d.max())}"
                )

                try:
                    # Create base results dataset with explicit dimensions
                    results = xr.Dataset(
                        {
                            "footprint_climatology": self.fclim_2d,  # Use DataArray directly
                            "domain_x": ("x", self.x),
                            "domain_y": ("y", self.y),
                        }
                    )

                    if hasattr(self, "f_2d") and self.f_2d is not None:
                        if "time" in self.f_2d.dims:
                            results["footprint_2d"] = self.f_2d
                        else:
                            self.logger.warning("f_2d missing time dimension")

                    # Add source areas if calculated
                    if hasattr(self, "source_areas") and self.source_areas is not None:
                        for r_level, area_dict in self.source_areas.items():
                            for key, value in area_dict.items():
                                if isinstance(value, xr.DataArray):
                                    results[f"{r_level}_{key}"] = value

                    self.logger.info("FFP model calculations completed successfully")
                    return results

                except Exception as e:
                    self.logger.error(f"Error creating results dataset: {str(e)}")
                    raise

        except Exception as e:
            self.logger.error(f"Error in FFP calculations: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)  # Add full traceback
            raise

        self.logger.info("FFP model calculations completed")
        return None

    def save_results(self, filename: str):
        """
        Save model results to a netCDF file.

        Args:
            filename: Path where the results should be saved
        """
        if getattr(self, "f_2d", None) is None or getattr(self, "fclim_2d", None) is None:
            raise RuntimeError("No results to save. Did you call run() first?")

        results = xr.Dataset(
            {
                "footprint_2d": self.f_2d,  # expect dims (time, x, y) or similar
                "footprint_climatology": self.fclim_2d,  # dims (x, y)
                "domain_x": xr.DataArray(self.x, dims=["x"]),
                "domain_y": xr.DataArray(self.y, dims=["y"]),
            }
        )

        # Store parameters as attributes (NetCDF-safe)
        results.attrs.update(
            {
                "zm": float(self.df["zm"].iloc[0]),
                "z0": float(self.df["z0"].iloc[0]),
                "atm_bound_height": float(self.df["h"].iloc[0]),
            }
        )

        results.to_netcdf(filename)
        self.logger.info(f"Results saved to {filename}")
