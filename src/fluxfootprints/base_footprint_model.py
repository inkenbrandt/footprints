# src/fluxfootprints/base_footprint_model.py
"""
base_footprint_model.py
========================
Base class defining standard interface for all footprint models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import xarray as xr
import logging


class BaseFootprintModel(ABC):
    """
    Abstract base class for flux footprint models.
    
    All footprint model implementations should inherit from this class
    and implement the required methods to ensure API consistency.
    
    Attributes
    ----------
    df : pandas.DataFrame
        Input meteorological data
    domain : list
        Spatial domain [xmin, xmax, ymin, ymax]
    dx, dy : float
        Grid resolution in x and y directions
    rs : list
        Source area fractions to compute
    logger : logging.Logger
        Logger instance
    x, y : numpy.ndarray
        Grid coordinate arrays
    f_2d : xarray.DataArray or None
        Time-resolved 2D footprint (time, x, y)
    fclim_2d : xarray.DataArray or None
        Climatological 2D footprint (x, y)
    """
    
    REQUIRED_COLUMNS = []  # Override in subclass
    
    def __init__(
        self,
        df: pd.DataFrame,
        domain: list = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: float = 10.0,
        dy: float = 10.0,
        rs: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        crop_height: float = 0.2,
        atm_bound_height: float = 2000.0,
        inst_height: float = 2.0,
        zm: Optional[float] = None,
        z0: Optional[float] = None,
        smooth_data: bool = True,
        verbosity: int = 2,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initialize base footprint model.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data with required meteorological columns
        domain : list
            [xmin, xmax, ymin, ymax] in meters
        dx, dy : float
            Grid spacing in meters
        rs : list
            Source area fractions (0-1) to compute
        zm : float, optional
            Effective measurement height above displacement height (m).
            When provided together with ``z0``, takes priority over
            ``crop_height`` / ``inst_height``.
        z0 : float, optional
            Aerodynamic roughness length (m).  Must be supplied together
            with ``zm`` when bypassing the ``crop_height`` derivation.
        crop_height : float
            Vegetation/canopy height (m). Used to derive ``zm`` and ``z0``
            when those are not supplied directly.
        atm_bound_height : float
            Atmospheric boundary layer height (m)
        inst_height : float or pandas.Series
            Instrument measurement height (m). Used to derive ``zm`` when
            ``zm`` is not supplied directly. Pass a ``pd.Series`` with the
            same DatetimeIndex as ``df`` to support height changes throughout
            the season (e.g. IRGASON repositioned mid-season).
        smooth_data : bool
            Apply smoothing to output
        verbosity : int
            Logging level (0=silent, 2=debug)
        logger : logging.Logger
            Custom logger instance
        """
        self.df = df.copy()
        self.domain = domain
        self.dx = float(dx)
        self.dy = float(dy)
        self.rs = rs
        self.crop_height = crop_height
        self.atm_bound_height = atm_bound_height
        self.inst_height = inst_height
        self.zm = zm
        self.z0 = z0
        self.smooth_data = smooth_data
        self.verbosity = int(verbosity)
        
        # Initialize logger
        self.logger = logger or self._setup_logger()
        
        # Initialize coordinate arrays
        self.x = None
        self.y = None
        
        # Initialize output arrays
        self.f_2d = None
        self.fclim_2d = None
        self.results = None
        
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if self.verbosity > 1 else logging.WARNING)
        return logger
    
    @abstractmethod
    def _validate_input_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input DataFrame.
        
        Must check for required columns and handle missing values.
        Should return cleaned DataFrame.
        """
        pass
    
    @abstractmethod
    def run(self, return_result: bool = True) -> Optional[xr.Dataset]:
        """
        Execute footprint calculation.
        
        Parameters
        ----------
        return_result : bool
            If True, return xarray Dataset with results
            
        Returns
        -------
        xarray.Dataset or None
            Dataset containing footprint_climatology (x, y),
            optionally footprint_2d (time, x, y), and metadata
        """
        pass
    
    def get_footprint_climatology(self) -> xr.DataArray:
        """
        Return the climatological footprint as xarray DataArray.
        
        Returns
        -------
        xarray.DataArray
            2D footprint climatology with dims (x, y)
        """
        if self.fclim_2d is None:
            raise RuntimeError("Model has not been run. Call run() first.")
        return self.fclim_2d
    
    def get_footprint_timeseries(self) -> Optional[xr.DataArray]:
        """
        Return time-resolved footprint if available.
        
        Returns
        -------
        xarray.DataArray or None
            3D footprint with dims (time, x, y) if available
        """
        return self.f_2d
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return x and y coordinate arrays.
        
        Returns
        -------
        x, y : numpy.ndarray
            1D coordinate arrays in meters
        """
        if self.x is None or self.y is None:
            raise RuntimeError("Coordinates not initialized. Call run() first.")
        return self.x, self.y
    
    def get_results(self) -> xr.Dataset:
        """
        Return complete results as xarray Dataset.
        
        Returns
        -------
        xarray.Dataset
            Dataset containing all footprint outputs and metadata
        """
        if self.results is None:
            raise RuntimeError("No results available. Call run() first.")
        return self.results
    
    def to_netcdf(self, filepath: str) -> None:
        """
        Save results to netCDF file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        """
        results = self.get_results()
        results.to_netcdf(filepath)
        self.logger.info(f"Results saved to {filepath}")