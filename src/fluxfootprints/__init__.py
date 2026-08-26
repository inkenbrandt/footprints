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

from .openet_masking import (
    apply_openet_mask,
    mask_footprint_dataarray,
    mask_summaries,
    mask_rasters_geotiff,
    index_openet_rasters,
    openet_mask_on_grid,
    footprint_grid_geometry,
    parse_raster_date,
    MaskedFootprint,
    GridGeometry,
)
from .openet_raster_diagnostics import mask_rasters_geotiff_with_diagnostics

from .footprint_animation import (
    FootprintAnimator,
    animate_footprint,
    resample_footprints,
    resolve_freq,
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
    "ffp_climatology_new",
    # Config / data loading
    "load_config",
    "load_amf_df",
    "compute_aerodynamic_params",
    # Climatology workflow
    "build_climatology",
    "summarize_periods",
    # Export helpers
    "export_contours_gpkg",
    "export_rasters_geotiff",
    "export_contour_stats_csv",
    # OpenET masking
    "apply_openet_mask",
    "mask_footprint_dataarray",
    "mask_summaries",
    "mask_rasters_geotiff",
    "mask_rasters_geotiff_with_diagnostics",
    "index_openet_rasters",
    "openet_mask_on_grid",
    "footprint_grid_geometry",
    "parse_raster_date",
    "MaskedFootprint",
    "GridGeometry",
    # Animation
    "FootprintAnimator",
    "animate_footprint",
    "resample_footprints",
    "resolve_freq",
    # Fetch / geospatial tools
    "polar_to_cartesian_dataframe",
    "aggregate_to_daily_centroid",
    "generate_density_raster",
    "concat_fetch_gdf",
    # NLDAS retrieval
    "call_nldas_time_series",
    "parse_nldas_csv",
    "fetch_nldas_forcing_dataset",
    # Kormann-Meixner analytical model
    "analytical_power_law_parameters",
    "length_scale_xi",
    "crosswind_integrated_footprint",
    "footprint_2d",
    "footprint_at_points",
    "cumulative_fetch",
    "effective_fetch",
    # Lagrangian stochastic model
    "LSFootprintConfig",
    "BackwardLSModel",
    "log_wind_profile",
    "sigma_w",
    "sigma_v",
    "lagrangian_timescale",
    # Wang & Davis analytical model
    "wang2006_fy",
    "reconstruct_gaussian_2d",
    # Constants
    "KAPPA",
    "__version__",
]

__version__ = "0.4.0"
