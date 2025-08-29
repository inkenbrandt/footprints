from typing import Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from scipy.ndimage import gaussian_filter
import xarray


class FFPclim:
    """
    Initialize the FFPclim class for computing flux footprint climatologies.

    Parameters
    ----------
    df : pandas.DataFrame
        Input time series data containing meteorological parameters and flux tower metadata.
    domain : np.ndarray, optional
        Spatial domain extents as [xmin, xmax, ymin, ymax]. Default is [-1000, 1000, -1000, 1000].
    dx : int, optional
        Grid resolution in the x-direction (meters). Default is 2.
    dy : int, optional
        Grid resolution in the y-direction (meters). Default is 2.
    nx : int, optional
        Number of grid cells in x-direction. Default is 1000.
    ny : int, optional
        Number of grid cells in y-direction. Default is 1000.
    rs : list or np.ndarray, optional
        Source area fractions (0–1) for which contour levels should be derived. Default is [0.1 to 0.8].
    crop_height : float, optional
        Canopy or vegetation height in meters. Default is 0.2.
    atm_bound_height : float, optional
        Atmospheric boundary layer height in meters. Default is 2000.0.
    inst_height : float, optional
        Instrument measurement height in meters. Default is 2.0.
    rslayer : bool, optional
        Whether to allow footprint calculation within the roughness sublayer. Default is False.
    smooth_data : bool, optional
        Whether to apply Gaussian smoothing to footprint climatology. Default is True.
    crop : bool, optional
        Whether to crop output to contour extent. Default is False.
    verbosity : int, optional
        Logging verbosity level (0: silent, 1–5: increasing detail). Default is 2.
    fig : bool, optional
        Whether to generate diagnostic plots. Default is False.
    logger : logging.Logger, optional
        Logger for status messages and error handling.
    time : optional
        Optional timestamp or index metadata (not actively used).
    crs : int, optional
        Coordinate reference system identifier. Placeholder for geospatial integration.
    station_x : float, optional
        X-coordinate of the measurement station. Placeholder for geospatial integration.
    station_y : float, optional
        Y-coordinate of the measurement station. Placeholder for geospatial integration.
    **kwargs : dict
        Additional keyword arguments (e.g., "show_heatmap") that control downstream processing.

    Notes
    -----
    This initializer configures all parameters and performs initial checks, preprocessing,
    domain definition, and conversion of input data to xarray format. Key constants and
    lookup tables are also initialized for use in footprint modeling.

    References
    ----------
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: n.kljun@swansea.ac.uk

    """

    def __init__(
        self,
        df: pd.DataFrame,
        domain: np.ndarray = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: int = 2,
        dy: int = 2,
        nx: int = 1000,
        ny: int = 1000,
        rs: Union[list, np.ndarray] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        crop_height: float = 0.2,
        atm_bound_height: float = 2000.0,
        inst_height: float = 2.0,
        rslayer: bool = False,
        smooth_data: bool = True,
        crop: bool = False,
        verbosity: int = 2,
        fig: bool = False,
        logger=None,
        time=None,
        crs: int = None,
        station_x: float = None,
        station_y: float = None,
        **kwargs,
    ):

        self.df = df

        # Model parameters
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.x = np.linspace(self.xmin, self.xmax, nx + 1)
        self.y = np.linspace(self.ymin, self.ymax, ny + 1)

        self.rotated_theta = None

        self.a = 1.4524
        self.b = -1.9914
        self.c = 1.4622
        self.d = 0.1359
        self.ac = 2.17
        self.bc = 1.66
        self.cc = 20.0
        self.oln = 5000.0  # limit to L for neutral scaling
        self.k = 0.4  # von Karman
        # ===========================================================================
        # Get kwargs
        self.show_heatmap = kwargs.get("show_heatmap", True)

        # ===========================================================================
        # Input check
        self.flag_err = 0

        self.dx = dx
        self.dy = dy

        self.rs = rs
        self.rslayer = rslayer
        self.smooth_data = smooth_data
        self.crop = crop
        self.verbosity = verbosity
        self.fig = fig

        self.time = None  # defined later after dropping na values
        self.ts_len = None  # defined by len of dropped df

        self.f_2d = None

        self.logger = logger

        if self.verbosity < 2:
            self.logger.setLevel(logging.INFO)
        elif self.verbosity == 3:
            self.logger.setLevel(logging.WARNING)
        elif self.verbosity == 4:
            self.logger.setLevel(logging.ERROR)
        elif self.verbosity == 5:
            self.logger.setLevel(logging.CRITICAL)
        else:
            self.logger.setLevel(logging.DEBUG)

        if "crop_height" in df.columns:
            h_c = df["crop_height"]
        else:
            h_c = crop_height

        if "atm_bound_height" in df.columns:
            h_s = df["atm_bound_height"]
        else:
            h_s = atm_bound_height

        if "inst_height" in df.columns:
            zm_s = df["inst_height"]
        else:
            zm_s = inst_height

        self.prep_df_fields(
            h_c=h_c,
            d_h=None,
            zm_s=zm_s,
            h_s=h_s,
        )
        self.define_domain()
        self.create_xr_dataset()

    def prep_df_fields(
        self,
        h_c=0.2,
        d_h=None,
        zm_s=2.0,
        h_s=2000.0,
    ):
        """
        Prepare and validate required DataFrame fields for footprint modeling.

        Parameters
        ----------
        h_c : float, optional
            Crop or canopy height in meters. Default is 0.2 m.
        d_h : float or None, optional
            Displacement height in meters. If None, estimated from h_c.
        zm_s : float, optional
            Instrument measurement height in meters. Default is 2.0 m.
        h_s : float, optional
            Atmospheric boundary layer height in meters. Default is 2000.0 m.

        Raises
        ------
        ValueError
            If any required fields are missing or input values fail validation.

        Notes
        -----
        This function estimates roughness and displacement height if not provided,
        renames required columns, performs range checks, and drops invalid rows.
        """

        if d_h is None:
            d_h = 10 ** (0.979 * np.log10(h_c) - 0.154)

        self.df["zm"] = zm_s - d_h
        self.df["h_c"] = h_c
        self.df["z0"] = h_c * 0.123
        self.df["h"] = h_s

        self.df = self.df.rename(
            columns={
                "V_SIGMA": "sigmav",
                "USTAR": "ustar",
                "wd": "wind_dir",
                "MO_LENGTH": "ol",
                "ws": "umean",
            }
        )

        self.df["zm"] = np.where(self.df["zm"] <= 0.0, np.nan, self.df["zm"])
        self.df["h"] = np.where(self.df["h"] <= 10.0, np.nan, self.df["h"])
        self.df["zm"] = np.where(self.df["zm"] > self.df["h"], np.nan, self.df["zm"])
        self.df["sigmav"] = np.where(self.df["sigmav"] < 0.0, np.nan, self.df["sigmav"])
        self.df["ustar"] = np.where(self.df["ustar"] <= 0.1, np.nan, self.df["ustar"])

        self.df["wind_dir"] = np.where(
            self.df["wind_dir"] > 360.0, np.nan, self.df["wind_dir"]
        )
        self.df["wind_dir"] = np.where(
            self.df["wind_dir"] < 0.0, np.nan, self.df["wind_dir"]
        )

        # Check if all required fields are in the DataFrame
        all_present = np.all(
            np.isin(["ol", "sigmav", "ustar", "wind_dir"], self.df.columns)
        )
        if all_present:
            self.logger.debug("All required fields are present")
        else:
            self.raise_ffp_exception(1)

        self.df = self.df.dropna(subset=["sigmav", "wind_dir", "h", "ol"], how="any")
        self.ts_len = len(self.df)
        self.logger.debug(f"input len is {self.ts_len}")

    def raise_ffp_exception(self, code):
        """
        Raise an FFP-specific exception with descriptive message.

        Parameters
        ----------
        code : int
            Error code corresponding to a predefined FFP error condition.

        Raises
        ------
        ValueError
            If the code corresponds to a fatal error.

        Notes
        -----
        Error codes and messages correspond to input and processing validity checks
        related to flux footprint modeling requirements.
        """

        exceptions = {
            1: "At least one required parameter is missing. Check the inputs.",
            2: "zm (measurement height) must be larger than zero.",
            3: "z0 (roughness length) must be larger than zero.",
            4: "h (boundary layer height) must be larger than 10 m.",
            5: "zm (measurement height) must be smaller than h (boundary layer height).",
            6: "zm (measurement height) should be above the roughness sub-layer.",
            7: "zm/ol (measurement height to Obukhov length ratio) must be >= -15.5.",
            8: "sigmav (standard deviation of crosswind) must be larger than zero.",
            9: "ustar (friction velocity) must be >= 0.1.",
            10: "wind_dir (wind direction) must be in the range [0, 360].",
        }

        message = exceptions.get(code, "Unknown error code.")

        if self.verbosity > 0:
            print(f"Error {code}: {message}")
            self.logger.info(f"Error {code}: {message}")

        if code in [1, 4, 5, 7, 9, 10]:  # Fatal errors
            self.logger.error(f"Error {code}: {message}")
            raise ValueError(f"FFP Exception {code}: {message}")

    def define_domain(self):
        """
        Define the spatial domain and coordinate grids for footprint calculations.

        Notes
        -----
        This method constructs Cartesian and polar 2D grids using the domain
        extents and resolutions, and initializes the footprint climatology array.
        """
        # ===========================================================================
        # Create 2D grid
        self.xv, self.yv = np.meshgrid(self.x, self.y, indexing="xy")

        # Define physical domain in cartesian and polar coordinates
        self.logger.debug(f"x: {self.x}, y: {self.y}")
        # Polar coordinates
        # Set theta such that North is pointing upwards and angles increase clockwise
        # Polar coordinates
        self.rho = xarray.DataArray(
            np.sqrt(self.xv**2 + self.yv**2),
            dims=("x", "y"),
            coords={"x": self.x, "y": self.y},
        )
        self.theta = xarray.DataArray(
            np.arctan2(self.yv, self.xv),
            dims=("x", "y"),
            coords={"x": self.x, "y": self.y},
        )
        self.logger.debug(f"rho: {self.rho}, theta: {self.theta}")
        # Initialize raster for footprint climatology
        self.fclim_2d = xarray.zeros_like(self.rho)

        # ===========================================================================

    def create_xr_dataset(self):
        """
        Convert the internal DataFrame into an xarray.Dataset for time-indexed modeling.

        Notes
        -----
        This conversion facilitates vectorized xarray operations and assigns 'time'
        as the primary dimension for input variables.
        """
        # Time series inputs as an xarray.Dataset
        self.df.index.name = "time"
        self.ds = xarray.Dataset.from_dataframe(self.df)

    def calc_xr_footprint(self):
        """
        Calculate the time-resolved footprint using vectorized xarray operations.

        Notes
        -----
        This method applies the Kljun et al. (2015) parameterization in a grid-wise
        fashion using wind direction rotation, stability corrections, and dispersion
        assumptions. The result is accumulated into a footprint climatology field.
        """

        # Rotate coordinates into wind direction
        self.rotated_theta = self.theta - (self.ds["wind_dir"] * np.pi / 180.0)

        psi_cond = np.logical_and(self.oln > self.ds["ol"], self.ds["ol"] > 0)

        # Compute xstar_ci_dummy for all timestamps
        xx = (1.0 - 19.0 * self.ds["zm"] / self.ds["ol"]) ** 0.25

        psi_f = xarray.where(
            psi_cond,
            -5.3 * self.ds["zm"] / self.ds["ol"],
            np.log((1.0 + xx**2) / 2.0)
            + 2.0 * np.log((1.0 + xx) / 2.0)
            - 2.0 * np.arctan(xx)
            + np.pi / 2.0,
        )

        xstar_bottom = xarray.where(
            self.ds["z0"].isnull(),
            (self.ds["umean"] / self.ds["ustar"] * self.k),
            (np.log(self.ds["zm"] / self.ds["z0"]) - psi_f),
        )

        xstar_ci_dummy = xarray.where(
            (np.log(self.ds["zm"] / self.ds["z0"]) - psi_f) > 0,
            self.rho
            * np.cos(self.rotated_theta)
            / self.ds["zm"]
            * (1.0 - (self.ds["zm"] / self.ds["h"]))
            / xstar_bottom,
            0.0,
        )

        xstar_ci_dummy = xstar_ci_dummy.astype(float)
        # Mask invalid values
        px = xstar_ci_dummy > self.d

        # Compute fstar_ci_dummy and f_ci_dummy
        fstar_ci_dummy = xarray.where(
            px,
            self.a
            * (xstar_ci_dummy - self.d) ** self.b
            * np.exp(-self.c / (xstar_ci_dummy - self.d)),
            0.0,
        )

        f_ci_dummy = xarray.where(
            px,
            fstar_ci_dummy
            / self.ds["zm"]
            * (1.0 - (self.ds["zm"] / self.ds["h"]))
            / xstar_bottom,
            0.0,
        )

        # Calculate sigystar_dummy for valid points
        sigystar_dummy = xarray.where(
            px,
            self.ac
            * np.sqrt(
                self.bc
                * np.abs(xstar_ci_dummy) ** 2
                / (1.0 + self.cc * np.abs(xstar_ci_dummy))
            ),
            0.0,  # Default value for invalid points
        )

        self.ds["ol"] = xarray.where(self.ds["ol"] > self.oln, -1e6, self.ds["ol"])

        # Calculate scale_const in a vectorized way
        scale_const = xarray.where(
            self.ds["ol"] <= 0,
            1e-5 * abs(self.ds["zm"] / self.ds["ol"]) ** (-1) + 0.80,
            1e-5 * abs(self.ds["zm"] / self.ds["ol"]) ** (-1) + 0.55,
        )
        scale_const = xarray.where(scale_const > 1.0, 1.0, scale_const)

        # Calculate sigy_dummy
        sigy_dummy = xarray.where(
            px,
            sigystar_dummy
            / scale_const
            * self.ds["zm"]
            * self.ds["sigmav"]
            / self.ds["ustar"],
            0.0,  # Default value for invalid points
        )

        sigy_dummy = xarray.where(sigy_dummy <= 0.0, np.nan, sigy_dummy)

        # sig_cond = np.logical_or(sigy_dummy.isnull(), px, sigy_dummy == 0.0)

        # Calculate the footprint in real scale
        self.f_2d = xarray.where(
            sigy_dummy.isnull(),
            0.0,
            f_ci_dummy
            / (np.sqrt(2 * np.pi) * sigy_dummy)
            * np.exp(
                -((self.rho * np.sin(self.rotated_theta)) ** 2) / (2.0 * sigy_dummy**2)
            ),
        )

        # self.f_2d = xr.where(px, self.f_2d, 0.0)

        # Accumulate into footprint climatology raster
        self.fclim_2d = self.f_2d.sum(dim="time")

        # Apply smoothing if requested
        if self.smooth_data:
            self.f_2d = xarray.apply_ufunc(
                gaussian_filter,
                self.f_2d,
                kwargs={"sigma": 1.0},
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]],
            )

    def smooth_and_contour(self):
        """
        Compute smoothed and normalized footprint climatology and extract contour levels.

        Returns
        -------
        xarray.Dataset
            Dataset containing contour masks for each specified source area fraction.

        Notes
        -----
        Applies optional Gaussian smoothing and constructs cumulative distribution-based
        contour masks for the normalized footprint field.
        """

        # Ensure the footprint data is normalized
        self.ds["footprint"] = self.ds["footprint"] / self.ds["footprint"].sum(
            dim=("x", "y")
        )

        # Apply smoothing if requested
        if self.smooth_data:
            self.ds["footprint"] = xarray.apply_ufunc(
                gaussian_filter,
                self.ds["footprint"],
                kwargs={"sigma": 1.0},
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]],
            )

        # Calculate cumulative footprint and extract contours
        cumulative = self.ds["footprint"].cumsum(dim="x").cumsum(dim="y")

        contours = {r: cumulative.where(cumulative >= r).fillna(0) for r in self.rs}

        # Combine results into a dataset
        climatology = xarray.Dataset(
            {f"contour_{int(r * 100)}": data for r, data in contours.items()}
        )

        return climatology

    def run(self):
        """
        Execute the complete flux footprint climatology workflow.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'x_2d': 2D x-coordinate grid (np.ndarray)
            - 'y_2d': 2D y-coordinate grid (np.ndarray)
            - 'fclim_2d': Accumulated footprint climatology (xarray.DataArray)
            - 'f_2d': Time-resolved footprint raster (xarray.DataArray)
            - 'rs': List of source area fractions used (list)

        Notes
        -----
        This wrapper calls the core footprint calculation routine and returns results
        in a structured dictionary.
        """
        self.calc_xr_footprint()
        # self.smooth_and_contour()
        return {
            "x_2d": self.xv,
            "y_2d": self.yv,
            "fclim_2d": self.fclim_2d,
            "f_2d": self.f_2d,
            "rs": self.rs,
        }
