# footprints/__init__.py
from .base_footprint_model import BaseFootprintModel
from .improved_ffp import FFPModel
from .kormannmeixner_adapter import KormannMeixnerModel
from .ls_footprint_adapter import LSFootprintModelAdapter
from .wang_footprint_adapter import WangFootprintModel


from .ffp_xr import ffp_climatology_new  # type: ignore[import]

from .ffp_daily_monthly_helper import (
    load_config,
    load_amf_df,
    build_climatology,
    summarize_periods,
    export_contours_gpkg,
    export_rasters_geotiff,
    export_contour_stats_csv,
    compute_aerodynamic_params,
)

from .by_row_fetch_tools import (
    polar_to_cartesian_dataframe,
    aggregate_to_daily_centroid,
    generate_density_raster,
    concat_fetch_gdf,
)

from .nldas_read_functions import (
    call_nldas_time_series,
    parse_nldas_csv,
    fetch_nldas_forcing_dataset,
)

from .kormannmeixner import (  # type: ignore
    analytical_power_law_parameters,
    length_scale_xi,
    crosswind_integrated_footprint,
    footprint_2d,
    footprint_at_points,
    cumulative_fetch,
    effective_fetch,
    KAPPA,
)

from .ls_footprint_model import (  # noqa: E402
    KAPPA,
    LSFootprintConfig,
    BackwardLSModel,
    log_wind_profile,
    sigma_w,
    sigma_v,
    lagrangian_timescale,
)

from .wang_footprint import wang2006_fy, reconstruct_gaussian_2d

__all__ = [
    # Base interface
    "BaseFootprintModel",
    # Model implementations
    "FFPModel",
    "KormannMeixnerModel",
    "LSFootprintModelAdapter",
    "WangFootprintModel",
    # Helper functions
    "build_climatology",
    # ... rest ...
]

__version__ = "0.3.0"
