import logging
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import xarray as xr

from .base_footprint_model import BaseFootprintModel

class ffp_climatology_new(BaseFootprintModel):
    r"""
    Create a footprint‑climatology from a series of individual flux‑footprint
    estimates using the simple 2‑D parameterisation of Kljun et al. (2015).

    The routine

    1. computes a footprint for every time step,
    2. rotates each footprint into its observed wind direction,
    3. aggregates all footprints onto a common grid, and
    4. derives user‑requested source‑area contours (``rs``).

    The implementation is based on:

        Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2015).
        *A simple two‑dimensional parameterisation for flux‑footprint
        predictions (FFP)*. **Geoscientific Model Development, 8**, 3695‑3713.
        https://doi.org/10.5194/gmd‑8‑3695‑2015

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time series of micrometeorological inputs.
        Must contain the following required columns:

        zm : float or Series
            Measurement height above displacement height (i.e. z – d) [m].
        z0 : float or Series or None
            Surface roughness length [m].
        umean : float or Series
            Mean wind speed at zm [m s⁻¹].
        h : float or Series
            Boundary‑layer height [m].
        ol : float or Series
            Obukhov length [m].
        sigmav : float or Series
            Standard deviation of lateral velocity fluctuations [m s⁻¹].
        ustar : float or Series
            Friction velocity [m s⁻¹].
        wind_dir : float or Series
            Wind direction in degrees (0–360°, meteorological convention).

    domain : array_like of float, default=[-1000.0, 1000.0, -1000.0, 1000.0]
        Domain limits ``[xmin, xmax, ymin, ymax]`` [m].
    dx, dy : float, default=10.0
        Grid‑cell size in x and y directions [m].
    nx, ny : int, default=1000
        Number of grid cells in x and y. Ignored if `dx`/`dy` are supplied.
    rs : list or ndarray, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        Source‑area contour fractions (e.g., 0.8 for 80%).
    rslayer : bool, default=False
        If True, allow calculations when `zm` lies in the roughness sub‑layer (RS).
        Warning: Model is formally invalid within the RS.
    smooth_data : bool, default=True
        Apply a Gaussian convolution filter to smooth the footprint climatology.
    crop : bool, default=False
        Crop the output grid to the extent of the largest requested contour.
    verbosity : {0, 1, 2}, default=2
        Verbosity level: 0 = silent, 1 = fatal only, 2 = warning/info.
    fig : bool, default=False
        Plot an example footprint on screen.
    Returns
    -------
    FFP : dict
        Dictionary with keys (see below) carrying the climatology results.
    x_2d, y_2d : ndarray
        2‑D grids (mesh‑grids) of x and y coordinates [m].
    fclim_2d : ndarray
        Normalised footprint‑climatology values [m⁻²].
    rs : ndarray or None
        Echo of input `rs` in percent, or *None* when `rs` was *None*.
    fr : ndarray or None
        Footprint value at each `rs` contour.
    xr, yr : list[ndarray] or None
        x‑ and y‑coordinates of every `rs` contour line.
    n : int
        Number of individual footprints used in the aggregation.
    flag_err : {0, 1, 2, 3}
        Error/status flag:
        *0* no error, *1* fatal error, *2* some contours outside domain,
        *3* invalid input rows removed.

    Notes
    -----
    *Implemented by* N. Kljun & G. Fratini, originally in MATLAB,
    ported to Python 3.x (v 1.4, 11 Dec 2019).

    Examples
    --------
    >>> import pandas as pd
    >>> from fluxfootprints import ffp_climatology_new
    >>> # Prepare DataFrame with required columns
    >>> df = pd.DataFrame({
    ...     "zm": [2.5], "z0": [0.1], "umean": [3.2], "h": [800.0],
    ...     "ol": [-50.0], "sigmav": [0.4], "ustar": [0.25], "wind_dir": [180.0]
    ... })
    >>> model = ffp_climatology_new(df=df, dx=5.0, dy=5.0)
    >>> results = model.run()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        domain: np.ndarray = [-1000.0, 1000.0, -1000.0, 1000.0],
        dx: int = 10.0,
        dy: int = 10.0,
        nx: int = 1000,
        ny: int = 1000,
        rs: Union[list, np.ndarray] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        rslayer: bool = False,
        smooth_data: bool = True,
        crop: bool = False,
        verbosity: int = 2,
        fig: bool = False,
        logger=None,
        time=None,
        **kwargs,
    ):
        
        self.df = df.copy()

        super().__init__(
            df=df,
            domain=domain,
            dx=dx,
            dy=dy,
            rs=rs,
            smooth_data=smooth_data,
            verbosity=verbosity,
            logger=logger,
        )


        self.fclim_2d = None

        # Model parameters
        self.xmin, self.xmax, self.ymin, self.ymax = domain

        if dx is None and nx is not None:
            self.x = np.linspace(self.xmin, self.xmax, nx + 1)
            self.y = np.linspace(self.ymin, self.ymax, ny + 1)
        else:
            self.x = np.arange(self.xmin, self.xmax + dx, dx)
            self.y = np.arange(self.ymin, self.ymax + dy, dy)

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

        self.logger = self._ensure_logger(logger)

        if self.verbosity < 2:
            self.logger.setLevel(logging.INFO)
        elif self.verbosity < 3:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.DEBUG)

        self.prep_df_fields()

        self.define_domain()
        self.create_xr_dataset()

    def _ensure_logger(self, logger: Optional[logging.Logger]) -> logging.Logger:
        if isinstance(logger, logging.Logger):
            return logger
        lg = logging.getLogger("footprint_daily_summary")
        if not lg.handlers:
            lg.addHandler(logging.StreamHandler())
        lg.setLevel(logging.WARNING)
        return lg

    def _validate_input_df(self, df=None, config=None, required_vars=None):
        """Pass-through validation method required by BaseFootprintModel."""
        if df is not None and isinstance(df, pd.DataFrame):
            return df
        if hasattr(self, "df") and isinstance(self.df, pd.DataFrame):
            return self.df
        return pd.DataFrame()

    def prep_df_fields(self):
        """Sanitize DataFrame input range bounds and drop invalid timesteps.

        Assumes df already contains standardized column names from
        build_climatology.
        """
        required_fields = [
            "zm",
            "z0",
            "h",
            "ol",
            "sigmav",
            "ustar",
            "wind_dir",
            "umean",
        ]
        all_present = all(col in self.df.columns for col in required_fields)
        if not all_present:
            self.raise_ffp_exception(1)

        self.df["zm"] = np.where(self.df["zm"] <= 0.0, np.nan, self.df["zm"])
        self.df['z0'] = np.where(self.df['z0'] <= 0, np.nan, self.df["z0"])
        self.df["h"] = np.where(self.df["h"] <= 10.0, np.nan, self.df["h"])
        self.df["zm"] = np.where(self.df["zm"] > self.df["h"], np.nan, self.df["zm"])
        self.df["sigmav"] = np.where(self.df["sigmav"] < 0.0, np.nan, self.df["sigmav"])
        # Match the reference FFP filter (and improved_ffp): u* <= 0.1 m/s is
        # rejected (reference check_ffp_inputs uses ustar <= 0.1).
        self.df["ustar"] = np.where(self.df["ustar"] <= 0.1, np.nan, self.df["ustar"])

        self.df["wind_dir"] = np.where(
            self.df["wind_dir"] > 360.0, np.nan, self.df["wind_dir"]
        )
        self.df["wind_dir"] = np.where(
            self.df["wind_dir"] < 0.0, np.nan, self.df["wind_dir"]
        )

        self.df = self.df.dropna(subset=required_fields, how="any")
        self.ts_len = len(self.df)
        if not self.df.empty:
            self.raise_ffp_exception(2)
        self.logger.debug(f"input len is {self.ts_len}")

    def raise_ffp_exception(self, code):
        exceptions = {
            1: "At least one required parameter is missing. Check the inputs.",
            2: "At least one row in dataframe required after filtering bad values"
        }

        message = exceptions.get(code, "Unknown error code.")

        if self.verbosity > 0:
            print(f"Error {code}: {message}")
            self.logger.info(f"Error {code}: {message}")

        if code in [1, 2]:  # Fatal errors
            self.logger.error(f"Error {code}: {message}")
            raise ValueError(f"FFP Exception {code}: {message}")

    def define_domain(self):
        # ===========================================================================
        # Create 2D grid
        self.xv, self.yv = np.meshgrid(self.x, self.y, indexing="ij")

        # Define physical domain in cartesian and polar coordinates
        self.logger.debug(f"x: {self.x}, y: {self.y}")
        # Polar coordinates
        # Set theta such that North is pointing upwards and angles increase clockwise
        # Polar coordinates
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
        self.logger.debug(f"rho: {self.rho}, theta: {self.theta}")
        # Initialize raster for footprint climatology
        self.fclim_2d = xr.zeros_like(self.rho)

        # ===========================================================================

    def create_xr_dataset(self):
        # Time series inputs as an xarray.Dataset
        self.df.index.name = "time"
        self.ds = xr.Dataset.from_dataframe(self.df)

    def calc_xr_footprint(self):
        # Rotate coordinates into wind direction
        self.rotated_theta = self.theta - (self.ds["wind_dir"] * np.pi / 180.0)

        psi_cond = np.logical_and(self.oln > self.ds["ol"], self.ds["ol"] > 0)

        # Compute xstar_ci_dummy for all timestamps
        xx = (1.0 - 19.0 * self.ds["zm"] / self.ds["ol"]) ** 0.25

        psi_f = xr.where(
            psi_cond,
            -5.3 * self.ds["zm"] / self.ds["ol"],
            np.log((1.0 + xx**2) / 2.0)
            + 2.0 * np.log((1.0 + xx) / 2.0)
            - 2.0 * np.arctan(xx)
            + np.pi / 2.0,
        )

        xstar_bottom = xr.where(
            self.ds["z0"].isnull(),
            (self.ds["umean"] / self.ds["ustar"] * self.k),
            (np.log(self.ds["zm"] / self.ds["z0"]) - psi_f),
        )

        xstar_ci_dummy = xr.where(
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
        fstar_ci_dummy = xr.where(
            px,
            self.a
            * (xstar_ci_dummy - self.d) ** self.b
            * np.exp(-self.c / (xstar_ci_dummy - self.d)),
            0.0,
        )

        f_ci_dummy = xr.where(
            px,
            fstar_ci_dummy
            / self.ds["zm"]
            * (1.0 - (self.ds["zm"] / self.ds["h"]))
            / xstar_bottom,
            0.0,
        )

        # Calculate sigystar_dummy for valid points
        sigystar_dummy = xr.where(
            px,
            self.ac
            * np.sqrt(
                self.bc
                * np.abs(xstar_ci_dummy) ** 2
                / (1.0 + self.cc * np.abs(xstar_ci_dummy))
            ),
            0.0,  # Default value for invalid points
        )

        self.ds["ol"] = xr.where(np.abs(self.ds["ol"]) > self.oln, -1e6, self.ds["ol"])

        # Crosswind-spread scaling factor ps1 (Kljun et al. 2015, Sect. 3.2):
        #   ps1 = min(1, |zm/L|^-1 * 1e-5 + p),  p = 0.80 (L<=0), 0.55 (L>0)
        # i.e. p plus a near-neutral ramp (1e-5*|L/zm|) capped at 1. The |L|>oln
        # remap above sends near-neutral cases to ps1 -> 1. Using a bare 0.80/0.55
        # step (the previous code) omits the ramp/cap and yields footprints up to
        # ~1.8x too wide crosswind in near-neutral conditions.
        p = xr.where(self.ds["ol"] <= 0, 0.80, 0.55)
        scale_const = np.minimum(
            1.0, 1e-5 * np.abs(self.ds["ol"] / self.ds["zm"]) + p
        )

        # Calculate sigy_dummy
        sigy_dummy = xr.where(
            px,
            sigystar_dummy
            / scale_const
            * self.ds["zm"]
            * self.ds["sigmav"]
            / self.ds["ustar"],
            0.0,  # Default value for invalid points
        )

        # less than or equal to zero covers the px filter as well
        sigy_dummy = xr.where(sigy_dummy <= 0.0, np.nan, sigy_dummy)

        # sig_cond = np.logical_or(sigy_dummy.isnull(), px, sigy_dummy == 0.0)
        #

        # Calculate the footprint in real scale
        self.f_2d = xr.where(
            sigy_dummy.isnull(),
            0.0,
            f_ci_dummy
            / (np.sqrt(2 * np.pi) * sigy_dummy)
            * np.exp(
                -((self.rho * np.sin(self.rotated_theta)) ** 2) / (2.0 * sigy_dummy**2)
            ),
        )

        # Apply Kljun et al. (2015) Eq. 27 validity bounds — zero out timesteps
        # outside the parameterisation's stated applicability range so they do not
        # contribute weight to the climatology.
        stability_valid = (self.ds["zm"] / self.ds["ol"]) >= -15.5
        height_valid = (
            (self.ds["zm"] > 20.0 * self.ds["z0"])
            & (self.ds["zm"] < 0.8 * self.ds["h"])
        )
        valid_mask = stability_valid & height_valid
        if not self.rslayer:
            # Exclude in-RSL timesteps (zm ≤ n_rsl·h_rs ≈ 27.5·z0) when not in
            # RSL mode; the parameterisation is formally invalid below this height.
            valid_mask = valid_mask & (self.ds["zm"] > 27.5 * self.ds["z0"])

        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            self.logger.warning(
                f"{n_invalid}/{self.ts_len} timesteps excluded: outside "
                "Kljun et al. (2015) validity bounds (Eq. 27)"
            )

        self.f_2d = xr.where(valid_mask, self.f_2d, 0.0)

        # Reproduce the original Kljun et al. (2015) aggregation: sum the
        # real-scale footprint densities [m^-2] over time and divide by the
        # number of valid footprints. The original does NOT renormalise each
        # footprint, so footprints whose mass partly falls outside the domain
        # contribute only their captured fraction. (The previous code divided
        # each timestep by its grid-cell *count* sum, which both dropped the
        # dx*dy area weighting -- yielding a non-density climatology off by a
        # factor dx*dy -- and gave every timestep equal weight regardless of
        # captured mass.)
        fp_sums = self.f_2d.sum(dim=("x", "y"))
        valid_count = int((fp_sums > 0).sum())
        self.fclim_2d = self.f_2d.sum(dim="time") / max(valid_count, 1)

        # Apply smoothing if requested
        if self.smooth_data:
            self.f_2d = xr.apply_ufunc(
                gaussian_filter,
                self.f_2d,
                kwargs={"sigma": 1.0},
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]],
            )
            self.fclim_2d = xr.apply_ufunc(
                gaussian_filter,
                self.fclim_2d,
                kwargs={"sigma": 1.0},
                input_core_dims=[["x", "y"]],
                output_core_dims=[["x", "y"]],
            )

    def run(self, return_result: bool = True):

        self.calc_xr_footprint()

        if return_result:
            results = xr.Dataset(
                        {
                            "footprint_climatology": self.fclim_2d,  # Use DataArray directly
                            "domain_x": ("x", self.x),
                            "domain_y": ("y", self.y),
                        }
                    )
            return results