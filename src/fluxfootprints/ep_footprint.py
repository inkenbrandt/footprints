import numpy as np
import scipy
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Footprint:
    """Class to store footprint calculation results"""

    peak: float  # Location of peak influence
    offset: float  # Location of 1% contribution
    x10: float  # Distance including 10% of footprint
    x30: float  # Distance including 30% of footprint
    x50: float  # Distance including 50% of footprint
    x70: float  # Distance including 70% of footprint
    x80: float  # Distance including 80% of footprint
    x90: float  # Distance including 90% of footprint

    @classmethod
    def error(cls):
        """Create an error footprint with all values set to -9999"""
        return cls(*[-9999.0] * 8)


# Constants
ERROR = -9999.0
KJ_US_MIN = 0.2  # minimum ustar for Kljun model
KJ_ZL_MIN = -200.0  # minimum zL for Kljun model
KJ_ZL_MAX = 1.0  # maximum zL for Kljun model

# Pre-calculated L values for Kljun model (from original code)
L_VALUES = np.array(
    [
        0.000000,
        0.302000,
        0.368000,
        0.414000,
        0.450000,
        0.482000,
        0.510000,
        0.536000,
        0.560000,
        0.579999,
        0.601999,
        0.621999,
        0.639999,
        0.657998,
        0.675998,
        0.691998,
        0.709998,
        0.725998,
        0.741997,
        0.755997,
        0.771997,
        0.785997,
        0.801997,
        0.815996,
        0.829996,
        0.843996,
        0.857996,
        0.871996,
        0.885995,
        0.899995,
        0.911995,
        0.925995,
        0.939995,
        0.953995,
        0.965994,
        0.979994,
        0.993994,
        1.005994,
        1.019994,
        1.033994,
        1.045993,
        1.059993,
        1.073993,
        1.085993,
        1.099993,
        1.113993,
        1.127992,
        1.141992,
        1.155992,
        1.169992,
        1.183992,
        1.197991,
        1.211991,
        1.225991,
        1.239991,
        1.253991,
        1.269991,
        1.283990,
        1.299990,
        1.315990,
        1.329990,
        1.345990,
        1.361989,
        1.379989,
        1.395989,
        1.411989,
        1.429989,
        1.447988,
        1.465988,
        1.483988,
        1.501988,
        1.521987,
        1.539987,
        1.559987,
        1.581987,
        1.601986,
        1.623986,
        1.647986,
        1.669985,
        1.693985,
        1.719985,
        1.745984,
        1.773984,
        1.801984,
        1.831983,
        1.863983,
        1.895983,
        1.931982,
        1.969982,
        2.009982,
        2.053984,
        2.101986,
        2.153988,
        2.211991,
        2.279994,
        2.355998,
    ]
)


def handle_footprint(
    var_w: float,
    ustar: float,
    zL: float,
    wind_speed: float,
    MO_length: float,
    sonic_height: float,
    disp_height: float,
    rough_length: float,
    foot_model: str = "kljun_04",
) -> Footprint:
    """
    Dispatch function for selecting and computing flux footprint models.

    This function selects an appropriate footprint model based on user input
    and input parameter validity. It supports three footprint models: Kljun et al. (2004),
    Kormann and Meixner (2001), and Hsieh et al. (2000). Automatically switches to a
    fallback model if input constraints are violated.

    Parameters
    ----------
    var_w : float
        Variance of vertical wind speed [m² s⁻²].
    ustar : float
        Friction velocity [m s⁻¹].
    zL : float
        Stability parameter (z / L) [-].
    wind_speed : float
        Mean horizontal wind speed [m s⁻¹].
    MO_length : float
        Monin-Obukhov length [m].
    sonic_height : float
        Height of the sonic anemometer above ground level [m].
    disp_height : float
        Displacement height [m].
    rough_length : float
        Surface roughness length [m].
    foot_model : str, optional
        Footprint model to use: one of {"kljun_04", "kormann_meixner_01", "hsieh_00"}.
        Defaults to "kljun_04".

    Returns
    -------
    Footprint
        A dataclass containing footprint distance metrics. Returns an error footprint
        if the selected model or parameters are invalid.

    Notes
    -----
    If the `"kljun_04"` model is selected but input values are outside its
    valid range, the function automatically switches to `"kormann_meixner_01"`.

    See Also
    --------
    kljun_04 : Analytical footprint model by Kljun et al. (2004).
    kormann_meixner_01 : Footprint model by Kormann and Meixner (2001).
    hsieh_00 : Footprint model by Hsieh et al. (2000).
    """
    if foot_model == "none":
        return Footprint.error()

    # If Kljun model conditions not met, switch to Kormann and Meixner
    if foot_model == "kljun" and (
        var_w <= 0.0
        or ustar < KJ_US_MIN
        or zL < KJ_ZL_MIN
        or zL > KJ_ZL_MAX
        or sonic_height < 1.0
    ):
        foot_model = "kormann_meixner"

    # Calculate std_w
    std_w = np.sqrt(var_w) if var_w >= 0.0 else ERROR

    if foot_model == "kljun":
        return kljun(std_w, ustar, zL, sonic_height, disp_height, rough_length)
    elif foot_model == "kormann_meixner":
        return kormann_meixner(ustar, zL, wind_speed, sonic_height, disp_height)
    elif foot_model == "hsieh":
        return hsieh(MO_length, sonic_height, disp_height, rough_length)
    else:
        return Footprint.error()


def kljun(
    std_w: float,
    ustar: float,
    zL: float,
    sonic_height: float,
    disp_height: float,
    rough_length: float,
) -> Footprint:
    """
    Estimate footprint characteristics using the Kljun et al. (2004) parameterization.

    This function computes footprint distances corresponding to peak and cumulative
    source area contributions (1–90%) using the analytical model from Kljun et al. (2004,
    Boundary-Layer Meteorology). It assumes steady-state, horizontally homogeneous
    turbulence and neutral to moderately stable or unstable atmospheric conditions.

    Parameters
    ----------
    std_w : float
        Standard deviation of vertical velocity fluctuations [m s⁻¹].
    ustar : float
        Friction velocity [m s⁻¹].
    zL : float
        Stability parameter (z / L), where L is the Obukhov length [-].
    sonic_height : float
        Height of the sonic anemometer above ground level [m].
    disp_height : float
        Displacement height (zero-plane displacement) [m].
    rough_length : float
        Surface roughness length [m].

    Returns
    -------
    Footprint
        A dataclass containing the peak location and distances at which 1%, 10%, 30%, 50%,
        70%, 80%, and 90% of the footprint is accumulated. Returns an error footprint
        if inputs are invalid.

    Notes
    -----
    The model is valid for:
    - 0.01 < z/L < 15
    - u* ≥ 0.21 m s⁻¹
    - z - d > 1.0 m
    - z₀ > 0

    References
    ----------
    Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2004).
    A simple parameterisation for flux footprint predictions.
    Boundary-Layer Meteorology, 112(3), 503–523. https://doi.org/10.1023/B:BOUN.0000030653.71031.96
    """
    # Initialize to error
    if std_w == ERROR or ustar < KJ_US_MIN or zL < KJ_ZL_MIN or zL > KJ_ZL_MAX:
        return Footprint.error()

    # Height above displacement height
    zm = sonic_height - disp_height
    if zm < 1.0 or rough_length <= 0.0:
        return Footprint.error()

    # Calculate parameters depending only on z0 (Eq. 13-16 in Kljun et al. 2004)
    af = 0.175
    bb = 3.418
    ac = 4.277
    ad = 1.685
    b = 3.69895

    a = af / (bb - np.log(rough_length))
    c = ac * (bb - np.log(rough_length))
    d = ad * (bb - np.log(rough_length))

    # Calculate footprint characteristics
    scaling = zm * (std_w / ustar) ** (-0.8)

    # Location of peak influence
    xstarmax = c - d
    peak = xstarmax * scaling

    # Calculate distances including increasing percentages of the footprint
    def calc_distance(L_value):
        xstar = L_value * c - d
        return xstar * scaling

    return Footprint(
        peak=peak,
        offset=calc_distance(L_VALUES[1]),  # 1% contribution
        x10=calc_distance(L_VALUES[10]),  # 10% contribution
        x30=calc_distance(L_VALUES[30]),  # 30% contribution
        x50=calc_distance(L_VALUES[50]),  # 50% contribution
        x70=calc_distance(L_VALUES[70]),  # 70% contribution
        x80=calc_distance(L_VALUES[80]),  # 80% contribution
        x90=calc_distance(L_VALUES[90]),  # 90% contribution
    )


def kormann_meixner(
    ustar: float, zL: float, wind_speed: float, sonic_height: float, disp_height: float
) -> Footprint:
    """
    Estimate footprint characteristics using the Kormann and Meixner (2001) analytical model.

    This function computes the location of peak flux contribution and source area distances
    corresponding to cumulative contributions of 1–90% using the model described by
    Kormann and Meixner (2001). It integrates a 1D crosswind-integrated footprint function
    derived from power-law profiles of wind speed and eddy diffusivity.

    Parameters
    ----------
    ustar : float
        Friction velocity [m s⁻¹].
    zL : float
        Stability parameter (z / L) [-].
    wind_speed : float
        Mean horizontal wind speed at measurement height [m s⁻¹].
    sonic_height : float
        Height of the sonic anemometer above ground level [m].
    disp_height : float
        Displacement height [m].

    Returns
    -------
    Footprint
        A dataclass containing the peak location and distances at which 1%, 10%, 30%, 50%,
        70%, 80%, and 90% of the footprint is accumulated.

    Notes
    -----
    - Applies Monin-Obukhov similarity theory for stability correction.
    - Assumes horizontally homogeneous surface conditions and power-law profiles for both
      eddy diffusivity and wind speed.
    - Integration is terminated at 90% cumulative contribution.
    - For unstable conditions (z/L < 0), non-linear stability corrections are applied via
      Paulson (1970) similarity functions.

    References
    ----------
    Kormann, R., & Meixner, F. X. (2001).
    An analytical footprint model for non-neutral stratification.
    Boundary-Layer Meteorology, 99(2), 207–224. https://doi.org/10.1023/A:1018991015119
    """
    # von Karman constant
    k = 0.41

    # Height above displacement height
    zm = sonic_height - disp_height

    # Calculate similarity relations (Paulson, 1970)
    if zL > 0:
        # Stable conditions
        phi_m = 1.0 + 5.0 * zL
        phi_c = phi_m
        psi_m = -5.0 * zL
    else:
        # Unstable conditions
        phi_m = (1.0 - 16.0 * zL) ** (-1.0 / 4.0)
        phi_c = (1.0 - 16.0 * zL) ** (-1.0 / 2.0)
        eta = (1.0 - 16.0 * zL) ** (1.0 / 4.0)
        psi_m = (
            2.0 * np.log((1.0 + eta) / 2.0)
            + np.log((1.0 + eta**2) / 2.0)
            - 2.0 * np.arctan(eta)
            + np.pi / 2.0
        )

    # Change sign to conform with K&M usage
    psi_m = -1.0 * psi_m

    # Calculate intermediate parameters
    # Exponent of the diffusivity power law
    if zL > 0:
        n = 1.0 / phi_m
    else:
        n = (1.0 - 24.0 * zL) / (1.0 - 16.0 * zL)

    # Proportionality constant of the diffusivity power law (Eqs. 11 and 32)
    key = k * ustar * zm / (phi_c * zm**n)

    # Exponent of the wind speed power law
    m = ustar * phi_m / (k * wind_speed)

    # Proportionality constant of the wind speed power law
    U = wind_speed / zm**m

    # Intermediate parameters
    r = 2.0 + m - n
    mu = (1.0 + m) / r
    zeta = U * zm**r / (r**2 * key)

    # Initialize footprint calculation variables
    distances = []
    integral = 0.0
    dx = 1.0  # Integration step

    # Arrays to store percentage contributions
    target_percentages = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    found_distances = [None] * len(target_percentages)

    # Calculate peak location
    x_peak = zeta / (1.0 + mu)

    # Integration loop
    for i in range(1, 10001):
        x = i * dx
        # Cross-wind integrated 1D function (gamma function approximation)
        contribution = (
            zeta**mu * np.exp(-zeta / x) / (x ** (1.0 + mu) * scipy.special.gamma(mu))
        )
        integral += contribution * dx

        # Check if we've reached each target percentage
        for j, target in enumerate(target_percentages):
            if found_distances[j] is None and integral > target:
                found_distances[j] = x

        # Exit if we've found all percentages
        if integral > 0.9:  # 90% is our maximum target
            break

    # Create footprint object
    return Footprint(
        peak=x_peak,
        offset=found_distances[0],  # 1% contribution
        x10=found_distances[1],  # 10% contribution
        x30=found_distances[2],  # 30% contribution
        x50=found_distances[3],  # 50% contribution
        x70=found_distances[4],  # 70% contribution
        x80=found_distances[5],  # 80% contribution
        x90=found_distances[6],  # 90% contribution
    )


def hsieh(
    MO_length: float, sonic_height: float, disp_height: float, rough_length: float
) -> Footprint:
    """
    Estimate footprint characteristics using the Hsieh et al. (2000) model.

    This function implements the analytical footprint model developed by
    Hsieh et al. (2000), based on Monin-Obukhov similarity theory and a
    stability-dependent formulation. It calculates the peak flux contribution
    distance and footprint extents based on integrated contributions.

    Parameters
    ----------
    MO_length : float
        Monin-Obukhov length [m].
    sonic_height : float
        Height of the sonic anemometer above ground level [m].
    disp_height : float
        Displacement height [m].
    rough_length : float
        Surface roughness length [m].

    Returns
    -------
    Footprint
        A dataclass containing the peak location and distances at which 1%, 10%, 30%, 50%,
        70%, 80%, and 90% of the footprint is accumulated.

    Notes
    -----
    - The model distinguishes between stable, unstable, and neutral atmospheric conditions
      using the stability parameter zL = zu / L.
    - `zu` is a derived scaling height incorporating roughness and log-profile assumptions.
    - Integration is performed over 1D footprint function values to estimate distance thresholds.

    References
    ----------
    Hsieh, C.-I., Katul, G., & Chi, T.-W. (2000).
    An analytical model for estimating the footprint of turbulent fluxes in
    thermally stratified atmospheric flows.
    Journal of Geophysical Research: Atmospheres, 105(D8), 9651–9664.
    https://doi.org/10.1029/2000JD900052
    """
    # von Karman constant
    k = 0.41

    # Calculate measurement height above displacement
    zm = sonic_height - disp_height

    # Calculate zu parameter (Eq. 8 in Hsieh et al. 2000)
    zu = zm * (np.log(zm / rough_length) - 1.0 + rough_length / zm)

    # Calculate stability parameter
    zL = zu / MO_length

    # Set parameters D and P based on stability conditions (Eq. 17)
    if abs(zL) < 0.04:
        # Neutral and near-neutral conditions
        D = 0.97
        P = 1.0
    elif zL < 0:
        # Unstable conditions
        D = 0.28
        P = 0.59
    else:
        # Stable conditions
        D = 2.44
        P = 1.33

    # Calculate peak distance (maximum source location)
    x_peak = D * zu**P * abs(MO_length) ** (1.0 - P) / (2.0 * k**2)

    # Initialize variables for integration
    dx = 5.0  # Integration step size
    target_percentages = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    found_distances = [None] * len(target_percentages)

    # Integration loop
    for i in range(1, 10001):
        x = i * dx

        # Calculate crosswind-integrated footprint function
        fact = D * zu**P * abs(MO_length) ** (1.0 - P) / (k**2 * x)
        value = np.exp(-fact)

        # Check if we've reached each target percentage
        for j, target in enumerate(target_percentages):
            if found_distances[j] is None and value > target:
                found_distances[j] = x

        # Exit if we've found all percentages
        if value > 0.9:  # 90% is our maximum target
            break

    # Create and return Footprint object
    return Footprint(
        peak=x_peak,
        offset=found_distances[0],  # 1% contribution
        x10=found_distances[1],  # 10% contribution
        x30=found_distances[2],  # 30% contribution
        x50=found_distances[3],  # 50% contribution
        x70=found_distances[4],  # 70% contribution
        x80=found_distances[5],  # 80% contribution
        x90=found_distances[6],  # 90% contribution
    )
