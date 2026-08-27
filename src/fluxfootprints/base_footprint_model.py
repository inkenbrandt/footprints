# src/fluxfootprints/base_footprint_model.py
"""
base_footprint_model.py
========================
Base class defining standard interface for all footprint models.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xarray as xr


def _source_weight_threshold(
    values: np.ndarray | xr.DataArray,
    cell_area: float,
    fraction: float,
) -> float:
    """
    Find the footprint density enclosing a fraction of the total source weight.

    Shared kernel behind :meth:`~fluxfootprints.FFPModel.get_source_area_contour`
    and :func:`~fluxfootprints.representativeness.contour_level_for_fraction`:
    the grid is sorted in descending order and ``value * cell_area`` is
    accumulated until `fraction` is reached, so the value returned is the
    density of the last cell inside the contour.

    Parameters
    ----------
    values : numpy.ndarray or xarray.DataArray
        Footprint weights as densities [m-2]. Flattened, so any shape works.
    cell_area : float
        Area of one grid cell, ``dx * dy`` [m2].
    fraction : float
        Source-weight fraction to enclose, in (0, 1).

    Returns
    -------
    float
        Footprint density [m-2] at the contour. Cells at or above it enclose at
        least `fraction` of the total source weight.

    Raises
    ------
    ValueError
        If `cell_area` is not positive and finite, `fraction` lies outside
        (0, 1), or the grid carries no positive source weight.

    Notes
    -----
    Non-finite and non-positive cells are dropped before sorting. NaNs would
    otherwise sort ahead of every real weight once the order is reversed and
    poison the cumulative sum, and the smallest *positive* density is the
    meaningful saturation point for a grid that holds less than `fraction` of
    the source weight -- as the package's climatologies, which integrate to the
    mean captured fraction rather than to 1, routinely do.
    """
    if not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError(f"cell_area must be positive and finite, got {cell_area!r}.")
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction!r}.")

    flat = np.asarray(values, dtype=float).ravel()
    positive = flat[np.isfinite(flat) & (flat > 0.0)]
    if positive.size == 0:
        raise ValueError(
            "The footprint carries no positive source weight, so no contour "
            "level is defined."
        )

    ordered = np.sort(positive)[::-1]
    enclosed = np.cumsum(ordered) * cell_area
    idx = min(int(np.searchsorted(enclosed, fraction)), ordered.size - 1)
    return float(ordered[idx])


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
        domain: list|None = None,
        dx: float = 10.0,
        dy: float = 10.0,
        rs: list|None = None,
        smooth_data: bool = True,
        verbosity: int = 2,
        logger: logging.Logger|None = None,
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
        smooth_data : bool
            Apply smoothing to output
        verbosity : int
            Logging level (0=silent, 2=debug)
        logger : logging.Logger
            Custom logger instance
        """
        if domain is None or len(domain) != 4:
            domain = [-1000.0, 1000.0, -1000.0, 1000.0]

        if rs is None or not isinstance(rs, list) or not all(0 <= r <= 1 for r in rs):
            rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        self.df = df.copy()
        self.domain = domain
        self.dx = float(dx)
        self.dy = float(dy)
        self.rs = rs
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
    def run(self, return_result: bool = True) -> xr.Dataset|None:
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
    
    def get_footprint_timeseries(self) -> xr.DataArray|None:
        """
        Return time-resolved footprint if available.
        
        Returns
        -------
        xarray.DataArray or None
            3D footprint with dims (time, x, y) if available
        """
        return self.f_2d
    
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
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